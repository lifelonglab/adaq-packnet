from __future__ import print_function

import collections

import numpy as np

import torch
import torch.nn as nn
import copy


class SparsePruner(object):
    """Performs pruning on the given model."""

    def __init__(self, model, prune_perc, previous_masks, train_bias, train_bn, dataset2idx, capacity=None):
        self.model = model
        
        self.prune_perc = prune_perc
        self.prune_perc_ = {}

        for module_idx, module in enumerate(self.model.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if prune_perc is not None:
                    self.prune_perc_[module_idx] = prune_perc[module_idx]
                #else:
                #    self.prune_perc_[module_idx] = prune_perc

        self.train_bias = train_bias
        self.train_bn = train_bn
        self.masks = {}

        self.current_masks = None
        self.previous_masks = previous_masks
        
        self.current_dataset_idx = dataset2idx #previous_masks[valid_key].max()
        self.masks[self.current_dataset_idx] = previous_masks
        self.capacity = capacity
   
    def set_model_copy():
       for module_idx, module in enumerate(self.model_copy.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                self.previous_masks[module_idx] = self.previous_masks[module_idx].cuda()        
                module.weight.data[self.previous_masks[module_idx].lt(self.current_dataset_idx)] = 0.0
                              

    def pruning_mask(self, weights, previous_mask, layer_idx, mask_=None, bit_width=None):
        """Ranks weights by magnitude. Sets all below kth to 0.
           Returns pruned mask.
        """
        # Select all prunable weights, ie. belonging to current dataset.
        previous_mask = previous_mask.cuda()
        
        
        #tensor = weights[previous_mask.eq(self.current_dataset_idx) | previous_mask.eq(0)]
        tensor = weights[(self.capacity[layer_idx] < 32-bit_width).cuda().byte()]
        abs_tensor = tensor.abs()

        #first train       

        if mask_ is None:
            
            if self.prune_perc_[layer_idx] > 0:
                cutoff_rank = round(self.prune_perc_[layer_idx] * tensor.numel())
                #cutoff_rank = round(self.prune_perc * tensor.numel())
                cutoff_value = abs_tensor.view(-1).cpu().kthvalue(cutoff_rank)[0]

                # Remove those weights which are below cutoff and belong to current
                # dataset that we are training for.
                remove_mask = weights.abs().le(cutoff_value.cuda()) * (self.capacity[layer_idx] < 32-bit_width).cuda().byte()
                #remove_mask = weights.abs().le(cutoff_value.cuda()) * previous_mask.eq(self.current_dataset_idx)

                # mask = 1 - remove_mask
                previous_mask[remove_mask.eq(1)] = 0
                previous_mask[(self.capacity[layer_idx]>=32-bit_width).cuda().byte()] = 0
                
        else:
  
            #previous_mask[torch.from_numpy(mask_)==0] = 0
            previous_mask[mask_==0] = 0
            #previous_mask[torch.from_numpy(mask_.astype(int))==0] = 0

        mask = previous_mask
        print('Layer #%d, pruned %d/%d (%.2f%%) (Total in layer: %d)' %
              (layer_idx, mask.eq(0).sum(), tensor.numel(),
               100 * mask.eq(0).sum() / tensor.numel(), weights.numel()))
        return mask

    
    def prune(self, bit_width):
        """Gets pruning mask for each layer, based on previous_masks.
           Sets the self.current_masks to the computed pruning masks.
        """
        print('Pruning for dataset idx: %d' % (self.current_dataset_idx))
        
        #assert not self.current_masks, 'Current mask is not empty? Pruning twice?'
        cmasks = copy.deepcopy(self.current_masks)
        self.current_masks = {}

        #print('Pruning each layer by removing %.2f%% of values' %
        #      (100 * self.prune_perc))

        counter = 0
  
        #for module_idx, module in enumerate(self.model_copy.modules()):
        for module_idx, module in enumerate(self.model.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if self.ticket_masks is not None:                      
                    mask = self.pruning_mask(module.weight.data, self.previous_masks[module_idx], module_idx, mask_=self.ticket_masks[module_idx], bit_width=bit_width)
                else:                 
                    mask = self.pruning_mask(module.weight.data, self.previous_masks[module_idx], module_idx, bit_width=bit_width)
                
                self.current_masks[module_idx] = mask.cuda()

                # Set pruned weights to 0.
                weight = module.weight.data
                weight[mask.eq(0)] = 0.0
                counter += 1

            #elif (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)) and counter == id:
            #    counter += 1                    
            #    self.current_masks[module_idx] = torch.zeros(module.weight.data.size()) + self.current_dataset_idx

    def make_grads_zero(self):
        """Sets grads of fixed weights to 0."""
        assert self.current_masks

        for module_idx, module in enumerate(self.model.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if self.ticket_masks is None:
                    layer_mask = self.current_masks[module_idx]

                    # Set grads of all weights not belonging to current dataset to 0.
                    if module.weight.grad is not None:
                        module.weight.grad.data[layer_mask.cuda().ne(self.current_dataset_idx)] = 0
                        if not self.train_bias:
                            # Biases are fixed.
                            if module.bias is not None:
                                module.bias.grad.data.fill_(0)
                else:
                    layer_mask = self.current_masks[module_idx]

                    # Set grads of all weights not belonging to current dataset to 0.
                    if module.weight.grad is not None:
                        module.weight.grad.data[layer_mask.cuda().ne(1.0)] = 0
                        if not self.train_bias:
                            # Biases are fixed.
                            if module.bias is not None:
                                module.bias.grad.data.fill_(0)                    

            elif 'BatchNorm' in str(type(module)):
                # Set grads of batchnorm params to 0.
                if not self.train_bn:                       
                    module.weight.grad.data.fill_(0)
                    module.bias.grad.data.fill_(0)
        
    def make_pruned_zero(self):
        """Makes pruned weights 0."""
        assert self.current_masks
        counter = 0

        for module_idx, module in enumerate(self.model.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear): 
                if self.ticket_masks is None:                
                    layer_mask = self.current_masks[module_idx]
                    module.weight.data[layer_mask.eq(0)] = 0.0 
                else:                                  
                    layer_mask = self.ticket_masks[module_idx]
                    module.weight.data[layer_mask.eq(0)] = 0.0
 

    def apply_mask(self, dataset_idx):
        """To be done to retrieve weights just for a particular dataset."""
       
        for module_idx, module in enumerate(self.model.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if self.ticket_masks is None:
                    weight = module.weight.data
                    mask = self.previous_masks[module_idx]
                    weight[mask.gt(dataset_idx)] = 0.0
                else:                  
                    weight = module.weight.data
                    mask = self.ticket_masks[module_idx]
                    weight[mask.ne(1.0)] = 0.0

        
    def restore_biases(self, biases):
        """Use the given biases to replace existing biases."""
        
        for module_idx, module in enumerate(self.model.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if module.bias is not None:
                    module.bias.data.copy_(biases[module_idx])
        


    def get_biases(self):
        """Gets a copy of the current biases."""
        biases = {}
        for module_idx, module in enumerate(self.model.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if module.bias is not None:
                    biases[module_idx] = module.bias.data.clone()
        return biases


    def make_weights_random(self):
        assert self.previous_masks

        #self.masks[self.current_dataset_idx] = self.previous_masks
        #self.previous_masks = self.current_masks
        counter = 0
        for module_idx, module in enumerate(self.model.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data

                mask = self.current_masks[module_idx]

                k = np.count_nonzero(mask.eq(self.current_dataset_idx).cpu().numpy().flatten())

                weight[mask.eq(self.current_dataset_idx)] = 1-2*torch.rand(k).cuda()

                #mask[mask.eq(0)] = self.current_dataset_idx
                
                #self.current_masks[module_idx] = mask
                counter += 1

        #self.previous_masks = self.current_masks


    def make_finetuning_mask(self, bit_width=None):
        """Turns previously pruned weights into trainable weights for
           current dataset.
        """
        assert self.previous_masks

        
        self.masks[self.current_dataset_idx] = self.previous_masks
        self.previous_masks = self.current_masks
        self.current_dataset_idx += 1
        counter = 0

        for module_idx, module in enumerate(self.model.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
               
                if self.ticket_masks is None:
                    mask = self.current_masks[module_idx]
                else:
                    mask = self.ticket_masks[module_idx]

                #k = np.count_nonzero(mask.eq(0).cpu().numpy().flatten())
                #weight[mask.eq(0)] = 1-2*torch.rand(k).cuda()
                k = np.count_nonzero(((self.capacity[module_idx]<32-bit_width).cuda().byte()).cpu().numpy().flatten())
                weight[(self.capacity[module_idx]<32-bit_width).cuda().byte()] = 1-2*torch.rand(k).cuda()


                #mask[mask.eq(0)] = self.current_dataset_idx
                mask[(self.capacity[module_idx]<32-bit_width).cuda().byte()] = self.current_dataset_idx
                
                self.current_masks[module_idx] = mask
                    
                counter += 1
                self.capacity[module_idx] += bit_width
        return counter