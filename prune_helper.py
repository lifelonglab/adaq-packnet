import numpy as np
import torch.nn as nn


def set_weights_by_mask(mask, net, share_mask=None):
    state_dict = net.state_dict()
    #pdb.set_trace()
    counter = 0
    for k, i in state_dict.items():  
        
        icpu = i.cpu()
        b = icpu.data.numpy()
        sz = icpu.data.numpy().shape

        if len(sz) > 1:

            mask_ = mask[str(counter)]            
            b = mask_ * b 

            if share_mask is not None:   
                share_mask_ = share_mask[str(counter)].cpu().numpy()
                b = share_mask_ * b

            i = Variable(torch.from_numpy(b))
            state_dict[k] = i

            counter = counter + 1
              
    net.load_state_dict(state_dict)


def init_population(model, nr, size, min, max, manager=None, current_masks=None, min_=0, max_=0, two_masks=False):
    population = {}
    mask_gradient = {}
    mask = {}
    mask__ = {}
    cm = {}
    weight_checkpoint = 1

    for i in range(nr):

        #models[str(i)] = copy.deepcopy(model)
        solution = []
        solution_ = []
        for j in range(size):
            solution.append(min + (max-min)*random.random())
            solution_.append(min_ + (max_-min_)*random.random())

        solution[2] = 0       
        solution_[2] = 0

        population[str(i)] = solution

        counter = 0
        mask_, mask_g, mask___ = {}, {}, {}
        current_masks_ = copy.deepcopy(current_masks)

        for module_idx, module in enumerate(manager.model.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):

                ids = current_masks[module_idx].eq(manager.pruner.current_dataset_idx) | current_masks[module_idx].eq(0)
                idx = (current_masks[module_idx] == manager.pruner.current_dataset_idx).nonzero()
             
                nr = idx.size(0)
                elem = np.random.choice(nr, int(solution[counter]*nr), replace=False)
                
                for j in range(len(elem)):
                    current_masks_[module_idx][idx[elem[j]][0]][idx[elem[j]][1]] = 0

                mask_[str(counter)] = copy.deepcopy(current_masks_[module_idx])
                mask_[str(counter)] = mask_[str(counter)].cpu().numpy() 
                mask_[str(counter)][mask_[str(counter)]>1] = 1
                

                if two_masks:
                    idx__ = current_masks[module_idx].lt(manager.pruner.current_dataset_idx) & current_masks[module_idx].gt(0)
                    idx_ = (idx__ > 0).nonzero()
                    

                    nr = idx_.size(0)
                    elem = np.random.choice(nr, int(solution_[counter]*nr), replace=False)

                    for j in range(len(elem)):
                        idx__[idx_[elem[j]][0]][idx_[elem[j]][1]] = 0

                    mask___[str(counter)] = copy.deepcopy(idx__)
                    
                counter += 1
          
        cm[str(i)] = current_masks_
        mask[str(i)] = mask_
        mask__[str(i)] = mask___

    return population, mask, cm, mask__


def compute_size(net):
    size = 0
    state_dict = net.state_dict()
    for k, v in state_dict.items():
        if len(v.size()) > 1:
            size += np.prod(v.size())     
    return size


def compute_average_sparsity_by_percentage(model, percentage):
    size = 0
    for module_idx, module in enumerate(model.modules()):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            sz = module.weight.data.size()
            if (len(sz) == 4):
                                
                #idxs = (masks[module_idx].flatten() == 1).nonzero()

                #import pdb
                #pdb.set_trace()
                #idxs = idxs.numpy()
                size += np.prod(sz)*percentage[module_idx]

            if (len(sz) == 2):
                #idxs = (masks[module_idx].flatten() == 1).nonzero()
                #idxs = idxs.numpy()
                size += np.prod(sz)*percentage[module_idx]
    return size


def compute_average_sparsity(model, masks):
    size = 0
    for module_idx, module in enumerate(model.modules()):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            sz = module.weight.data.size()
            if (len(sz) == 4):
                                
                idxs = (masks[module_idx].flatten() > 0).nonzero()

                #import pdb
                #pdb.set_trace()
                idxs = idxs.numpy()
                size += idxs[0].size

            if (len(sz) == 2):
                idxs = (masks[module_idx].flatten() > 0).nonzero()
                idxs = idxs.numpy()
                size += idxs[0].size
    return size