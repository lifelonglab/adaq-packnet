import torch.nn as nn
from pruner import SparsePruner
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
import torchnet as tnt
import json
import torch

class Manager(object):

    def __init__(self, args, model, previous_masks, dataset2idx, dataset2biases, prune_perc_per_layer=None, masks=None, capacity=None, bit_width=None):
        self.args = args
        self.cuda = args.cuda
        self.model = model
        self.dataset2idx = dataset2idx
        self.dataset2biases = dataset2biases
        self.batchnorms = {}

        self.criterion = nn.CrossEntropyLoss()

        self.pruner = SparsePruner(self.model, prune_perc_per_layer, previous_masks, self.args.train_biases, self.args.train_bn, dataset2idx, capacity=capacity)
        self.pruner.ticket_masks = masks
        self.bit_width = bit_width


    def eval(self, test_loader, dataset_idx, biases=None, copy_train=False, cv=None):
        """Performs evaluation."""
        if not self.args.disable_pruning_mask:
            
            self.pruner.apply_mask(dataset_idx, copy_train=copy_train)
        if biases is not None:
            self.pruner.restore_biases(biases, copy_train=copy_train)

        self.model.eval()

        error_meter = None

        print('Performing eval...')
        for batch, label in tqdm(test_loader, desc='Eval'):
          
            batch = batch.cuda();
            batch = Variable(batch, volatile=False)

            output = self.model(batch)

            #label -= 175
            # Init error meter.
            if error_meter is None:
                topk = [1]
                if output.size(1) > 5:
                    topk.append(5)
                error_meter = tnt.meter.ClassErrorMeter(topk=topk)
            error_meter.add(output.data, label)

        errors = error_meter.value()
        print('Error: ' + ', '.join('@%s=%.2f' %
                                    t for t in zip(topk, errors)))
        
        self.model.train()
        return errors


    def do_batch(self, optimizer, batch, label):
        """Runs model for one batch."""

        #if self.cuda:
        batch = batch.cuda()
        label = label.cuda()

        batch = Variable(batch)
        label = Variable(label)

        # Set grads to 0.
        self.model.zero_grad()
        
        # Do forward-backward.

        output = self.model(batch) 

        #label -= 175
        self.criterion(output, label).backward()
        
        # Set fixed param grads to 0.
        if not self.args.disable_pruning_mask:
            self.pruner.make_grads_zero()
            self.pruner.prune(bit_width=self.bit_width)

        optimizer.step()
        
        # Set pruned weights to 0.
        if not self.args.disable_pruning_mask:
            self.pruner.make_pruned_zero()

        #postpruning(net, percentage)
        #if prune:
        #    net = set_weights_by_mask(net, net)
            

    #def copy_batchnorm(self, model_1, model_2):
    #    for module_1, module_2 in zip(model_1.modules(), model_2.modules()):
    #        if isinstance(module_1, nn.BatchNorm2d):
    #            module_2.running_var = module_1.running_var
    #            module_2.running_mean = module_1.running_mean


    #def apply_batchnorm(self, dataset_idx):
    #    batchnorms = self.batchnorms[dataset_idx]
    #    for module_idx, module in zip(model.modules()):
    #        if isinstance(module, nn.BatchNorm2d):
    #            model.modules()[module_idx] = batchnorms[module_idx]

    def do_epoch(self, train_loader, epoch_idx, optimizer):
        """Trains model for one epoch."""
        
        for batch, label in tqdm(train_loader, desc='Epoch: %d ' % (epoch_idx)):
            self.do_batch(optimizer, batch, label)                         

    def save_model(self, epoch, best_accuracy, errors, directory, checkpoint_name, capacity=None, batch_norms=None):
        """Saves model to file."""
        base_model = self.model

        # Prepare the ckpt.        
        self.dataset2biases[self.args.dataset] = self.pruner.get_biases()
        
        self.batchnorms[self.pruner.current_dataset_idx] = {}

        for module_idx, module in enumerate(base_model.modules()):
            if isinstance(module, nn.BatchNorm2d):
                self.batchnorms[self.pruner.current_dataset_idx][module_idx] = module

        ckpt = {
            'args': self.args,
            'epoch': epoch,
            'accuracy': best_accuracy,
            'errors': errors,
            'dataset2idx': self.pruner.current_dataset_idx,
            'previous_masks': self.pruner.current_masks,
            'masks': self.pruner.masks,
            'model': base_model,
            'capacity': capacity,
            'batchnorms': batch_norms,
        }
        if self.args.train_biases:
            ckpt['dataset2biases'] = self.dataset2biases

        # Save to file.
        torch.save(ckpt, directory+checkpoint_name)


    def train(self, train_loader, test_loader, epochs, optimizer, save=True, directory='', filename='checkpoint.pt', best_accuracy=0.0, capacity=None):
        """Performs training."""
        best_accuracy = best_accuracy
        error_history = []

        if self.args.cuda:
            self.model = self.model.cuda()

        for epoch in range(epochs):
            print('Epoch: %d' % (epoch))
            
            self.model.train()
            
            self.do_epoch(train_loader, epoch, optimizer)
            errors = self.eval(test_loader, self.pruner.current_dataset_idx)
            error_history.append(errors)
            accuracy = 100 - errors[0]  # Top-1 accuracy. 

            # Save performance history and stats.
            with open(filename + '.json', 'w') as fout:
                json.dump({
                    'error_history': error_history,
                    'args': vars(self.args),
                }, fout)

            # Save best model, if required.
            if save and accuracy > best_accuracy:
                print('Best model so far, Accuracy: %0.2f%% -> %0.2f%%' %
                      (best_accuracy, accuracy))
                best_accuracy = accuracy
                #get batch norms
                self.save_model(epoch, best_accuracy, errors, directory, filename, capacity=capacity)

        print('Finished finetuning...')
        print('Best error/accuracy: %0.2f%%, %0.2f%%' %
              (100 - best_accuracy, best_accuracy))
        print('-' * 16)
        return best_accuracy


    def prune(self):
        """Perform pruning."""
        print('Pre-prune eval:')
        self.eval(self.pruner.current_dataset_idx)       

        self.pruner.prune()
        self.check(True)

        print('\nPost-prune eval:')
        errors = self.eval(self.pruner.current_dataset_idx)
        
        accuracy = 100 - errors[0]  # Top-1 accuracy.
        self.save_model(-1, accuracy, errors,
                        self.args.save_prefix + '_postprune')

        # Do final finetuning to improve results on pruned network.
        if self.args.post_prune_epochs:
            print('Doing some extra finetuning...')

            optimizer = optim.Adam(net.parameters(), lr=0.0005,betas=(0.9,0.999), eps=1e-08, weight_decay=0.00, amsgrad=False)
            #optimizer = optim.SGD(self.model.parameters(),
            #                      lr=self.args.lr, momentum=0.9,
            #                      weight_decay=self.args.weight_decay)
            
            best_accuracy = self.train(self.args.post_prune_epochs, optimizer, save=True,
                       savename=self.args.save_prefix, best_accuracy=accuracy)

        print('-' * 16)
        print('Pruning summary:')
        self.check(True)
        print('-' * 16)
        return best_accuracy


    def check(self, verbose=False):
        """Makes sure that the layers are pruned."""
        print('Checking...')
        for layer_idx, module in enumerate(self.model.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                num_params = weight.numel()
                num_zero = weight.view(-1).eq(0).sum()
                if verbose:
                    print('Layer #%d: Pruned %d/%d (%.2f%%)' %
                          (layer_idx, num_zero, num_params, 100 * num_zero / num_params))