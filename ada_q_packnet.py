from __future__ import division, print_function

import argparse
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
from torch.autograd import Variable
from tqdm import tqdm

import dataset
import networks as net
#import utils as utils
from pruner import SparsePruner
from data.load import get_multitask_experiment
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import copy
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms
import random
import os
import math

from prune_helper import compute_average_sparsity
from prune_helper import compute_average_sparsity_by_percentage
from prune_helper import compute_size
from prune_helper import get_batchnorms
from non_linear_quantization import vquant
from prune_helper import set_weights_by_mask, init_population
from prune_manager import Manager

warnings.filterwarnings("ignore")

FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--arch',
                   choices=['vgg16', 'vgg16bn', 'resnet50', 'densenet121', 'mlp', 'alexnet', 'resnet18', 'tiny'],
                   help='Architectures')
FLAGS.add_argument('--mode',
                   choices=['quantize', 'prune', 'quantize_and_prune', 'eval'],  
                   help='Run mode')
FLAGS.add_argument('--run_option', default='single_task',
                   choices=['single_task', 'all_tasks'],
                   help='Run option')
FLAGS.add_argument('--num_outputs', type=int, default=10,
                   help='Num outputs for dataset')

# Optimization options.
FLAGS.add_argument('--lr', type=float, default=0.001,
                   help='Learning rate')
FLAGS.add_argument('--lr_decay_every', type=int,
                   help='Step decay every this many epochs')
FLAGS.add_argument('--lr_decay_factor', type=float,
                   help='Multiply lr by this much every step of decay')
FLAGS.add_argument('--epochs', type=int, default=10,
                   help='Number of initial finetuning epochs')
FLAGS.add_argument('--search_epochs', type=int, default=30,
                   help='Number of initial finetuning epochs')
FLAGS.add_argument('--finetune_epochs', type=int,
                   help='Number of initial finetuning epochs')
FLAGS.add_argument('--batch_size', type=int, default=32,
                   help='Batch size')
FLAGS.add_argument('--weight_decay', type=float, default=5e-4,
                   help='Weight decay')

# Paths.
FLAGS.add_argument('--dataset', type=str, default='CIFAR100',
                   help='Name of dataset')
FLAGS.add_argument('--permutations', type=str, default='permutations.pt',
                   help='permMNIST permutations')
FLAGS.add_argument('--train_path', type=str, default='',
                   help='Location of train data')
FLAGS.add_argument('--val_path', type=str, default='',
                   help='Location of test data')
FLAGS.add_argument('--save_prefix', type=str, default='/home/ctonn/Work/checkpoints/',
                   help='Location to save model')
FLAGS.add_argument('--loadname', type=str, default='alex_init.pt',
                   help='Location to save model')
FLAGS.add_argument('--initname', type=str, default='alex_init.pt',
                   help='Location to init model')
FLAGS.add_argument('--checkpoint', type=str, default='alex_0.pt',
                   help='Name of the checkpoint file')

# Pruning options.
FLAGS.add_argument('--prune_method', type=str,
                   choices=['lottery_ticket', 'lottery_ticket_search', 'fine_tuning', 'dynamic_mask', 'dynamic_mask_search'],
                   help='Pruning method to use')
FLAGS.add_argument('--prune_strategy', type=str,
                   choices=['random', 'maximize_from_previous_task'],
                   help='Pruning strategy to use')
FLAGS.add_argument('--prune_perc_per_layer', type=float, default=0.6,
                   help='% of neurons to prune per layer')
FLAGS.add_argument('--disable_pruning_mask', action='store_true', default=False,
                   help='use masking or not')
FLAGS.add_argument('--train_biases', action='store_true', default=False,
                   help='use separate biases or not')
FLAGS.add_argument('--train_bn', action='store_true', default=False,
                   help='train batch norm or not')
FLAGS.add_argument('--population_size', type=int, default=2)

# Other.
FLAGS.add_argument('--quantization_method', type=str,
                   choices=['nonlinear'],
                   help='Pruning method to use')
FLAGS.add_argument('--bit_width', type=int, default=8)
FLAGS.add_argument('--cuda', action='store_true', default=False,
                   help='use CUDA')
FLAGS.add_argument('--init_dump', action='store_true', default=True,
                   help='Initial model dump.')
FLAGS.add_argument('--task_number', type=int, default=10)
FLAGS.add_argument('--task_id', type=int, default=0)
FLAGS.add_argument('--classes_per_task', type=int, default=10)
FLAGS.add_argument('--V_min', type=float, default=0.4)
FLAGS.add_argument('--V_max', type=float, default=0.7)
    

def set_weights_by_mask(mask, net):
    state_dict = net.state_dict()

    counter = 0
    for k, i in state_dict.items():

        icpu = i.cpu()
        b = icpu.data.numpy()
        sz = icpu.data.numpy().shape

        if len(sz) > 1:

            if counter == 7:
                mask_ = mask[str(counter)]
                b = mask_ * b

                print(float(np.count_nonzero(b))/float(b.size))

                i = Variable(torch.from_numpy(b))
                state_dict[k] = i

            counter = counter + 1

    net.load_state_dict(state_dict)
    return net


def init_mask(model, layers, percentage, previous_masks, task_id, capacity, bit_width):
    counter = 0
    state_dict = model.state_dict()
    mask_ = {}

    #for k, m in state_dict.items():
    for module_idx, module in enumerate(model.modules()):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            sz = module.weight.data.size()
            if (len(sz) == 4):
                
                #import pdb
                #pdb.set_trace()
                #idxs = (previous_masks[counter].flatten() == task_id+1).nonzero()
                idxs = (capacity[module_idx]<=32-bit_width[module_idx]).flatten().cuda().byte().nonzero()
                idxs = idxs.cpu().numpy()

                mask_[counter] = np.ones((sz[0], sz[1], sz[2], sz[3]))

                #elem = np.random.choice(mask_[counter].size, int(percentage[counter]*mask_[counter].size), replace=False)
                elem = np.random.choice(idxs.size, int(percentage[counter]*idxs.size), replace=False)
                mask_[counter] = mask_[counter].flatten()
                mask_[counter][idxs[elem]] = 0.0

                mask_[counter] = torch.from_numpy(np.reshape(mask_[counter], (sz[0], sz[1], sz[2], sz[3])))

            if (len(sz) == 2):
                #idxs = (previous_masks[counter].flatten() == task_id+1).nonzero()
                idxs = (capacity[module_idx]<=32-bit_width[module_idx]).flatten().cuda().byte().nonzero()
                idxs = idxs.cpu().numpy()

                mask_[counter] = np.ones((sz[0], sz[1]))

                #elem = np.random.choice(mask_[counter].size, int(percentage[counter]*mask_[counter].size), replace=False)
                elem = np.random.choice(idxs.size, int(percentage[counter]*idxs.size), replace=False)
                mask_[counter] = mask_[counter].flatten()

                mask_[counter][idxs[elem]] = 0.0
                mask_[counter] = torch.from_numpy(np.reshape(mask_[counter], (sz[0], sz[1])))
        counter = counter + 1
    return mask_


def prune_(net, layers, mask):
    state_dict = net.state_dict()
    counter = 0

    for k, i in state_dict.items():
    
        icpu = i.cpu()
        b = icpu.data.numpy()
        sz = icpu.data.numpy().shape

        if len(sz) > 1:  #and not 'mask' in k:
            if counter in layers:

                mk = mask[str(counter)]

                if True:
                    if len(sz) == 4:
                        for x in range(sz[0]):
                            for y in range(sz[1]):
                                for w in range(sz[2]):
                                    for z in range(sz[3]):
                                        if mk[x][y][w][z] == 0:
                                            b[x][y][w][z] = 0.0

                    if len(sz) == 2:
                        for x in range(sz[0]):
                            for y in range(sz[1]):
                                if mk[x][y] == 0:
                                    b[x][y] = 0.0

            #else:
            #    if len(sz) == 4:
            #        for x in range(sz[0]):
            #            for y in range(sz[1]):
            #                for w in range(sz[2]):
            #                    for z in range(sz[3]):
            #                        if mask[str(counter)][x][y][w][z] == 0:
            #                            b[x][y][w][z] = 0.0

            #    if len(sz) == 2:
            #        for x in range(sz[0]):
            #            for y in range(sz[1]):
            #                if mask[str(counter)][x][y] == 0:
            #                    b[x][y] = 0.0


            counter = counter + 1
            i = Variable(torch.from_numpy(b))
            state_dict[k] = i

    net.load_state_dict(state_dict)
    return mask


def postpruning(net, percentage):

    state_dict = net.state_dict()
    counter = 0
    for k, v in state_dict.items():
        
        if len(v.size()) > 1:
            abs_tensor = v.abs()
            cutoff_rank = round(percentage[counter] * v.numel())
            cutoff_value = abs_tensor.view(-1).cpu().kthvalue(cutoff_rank)[0]

            remove_mask = v.abs().le(cutoff_value.cuda()) #* previous_mask.eq(self.current_dataset_idx)

            # mask = 1 - remove_mask
            v[remove_mask.eq(1)] = 0
            
            state_dict[k] = v
            counter =+ 1

    net.load_state_dict(state_dict)


def init_dump(arch, args):
    """Dumps pretrained model in required format."""

    if arch == 'alexnet':
        model = net.AlexCifarNet() #net.AlexNet()
    elif arch == 'vgg16':
        model = net.VGG('VGG16')
    elif arch == 'vgg16bn':
        model = net.ModifiedVGG16BN()
    elif arch == 'mlp':
        model = net.MLP_(32, 10)
    elif arch == 'resnet50':
        model = net.ModifiedResNet()
    elif arch == 'resnet18':
        from torchvision.models import resnet18
        model = resnet18(pretrained = False)
        model.fc = nn.Linear(512, 10)

    elif arch == 'tiny':
        model = net.TinyNet(5, 40, False)
    elif arch == 'densenet121':
        model = net.ModifiedDenseNet()
    else:
        raise ValueError('Architecture type not supported.')

    previous_masks = {}
    masks = {}
    capacity = {}
    #bit_width = {}

    for module_idx, module in enumerate(model.modules()):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            mask = torch.ByteTensor(module.weight.data.size()).fill_(1)
            weights_capacity = torch.Tensor(module.weight.data.size()).fill_(0)

            if 'cuda' in module.weight.data.type():
                mask = mask.cuda()
            previous_masks[module_idx] = mask
            capacity[module_idx] = weights_capacity
    bit_width = args.bit_width
    
    torch.save({
        'dataset2idx': 1,
        'previous_masks': previous_masks,
        'model': model,
        'capacity': capacity,
        'bit_width': bit_width,
    }, args.save_prefix + args.initname) #% (arch))


def main():
    """Do stuff."""
    args = FLAGS.parse_args()
    if args.init_dump:
        init_dump(args.arch, args)
        return
    if args.prune_perc_per_layer <= 0:
        return


    ckpt = torch.load(args.save_prefix + args.loadname) 
    model = ckpt['model']
    model.cuda()
    capacity = ckpt['capacity']
    bit_width = ckpt['bit_width']
   
    previous_masks = ckpt['previous_masks']
    dataset2idx = ckpt['dataset2idx']
    #masks = ckpt['masks']
    if 'dataset2biases' in ckpt:
        dataset2biases = ckpt['dataset2biases']
    else:
        dataset2biases = {}

    #take batch norms from previous model
   
    if args.dataset == 'CIFAR100' or args.dataset == 'permMNIST' or args.dataset == 'splitMNIST':
        (train_datasets, test_datasets), config, classes_per_task = get_multitask_experiment(name=args.dataset, scenario='class', tasks=10, data_dir='./store/datasets', normalize=False, augment=False, verbose=True, exception=False, only_test=False)
        
    if args.dataset == 'permMNIST':
        permutations = torch.load(args.save_prefix + args.permutations)  

        train_datasets = [get_dataset('mnist', permutation=p) for p in permutations]
        test_datasets = [get_dataset('mnist', train=False, permutation=p) for p in permutations]

    if args.dataset == '5_datasets':
        # MNIST
        #train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        #test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        #trainloaders.append(torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True))
        #testloaders.append(torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=True))

        # FashionMNIST
        #train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        #test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        #trainloaders.append(torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True))
        #testloaders.append(torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=True))

        #CIFAR10
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        trainloaders.append(train_loader)
        testloaders.append(val_loader)
        
    if args.dataset == 'Imagenet100' or args.dataset == 'TinyImagenet':
        
        if args.dataset == 'Imagenet100':
            size = 224

        if args.dataset == 'TinyImagenet':
            size = 64

        traindir = os.path.join(args.train_path, 'train')
        valdir = os.path.join(args.val_path, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(traindir, transforms.Compose([transforms.RandomResizedCrop(size), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize,]))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True, sampler=train_sampler)

        if args.dataset == 'Imagenet100':
            val_dataset = datasets.ImageFolder(valdir, transforms.Compose([transforms.Resize(size+32), transforms.CenterCrop(size), transforms.ToTensor(), normalize,]))
        if args.dataset == 'TinyImagenet':
            val_dataset = datasets.ImageFolder(valdir, transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor(), normalize,]))

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
        
    if args.run_option == 'all_tasks':  
        for i in range(args.task_number):
            pass

    if args.run_option == 'single_task':
        #prepare data

        if args.dataset == 'Imagenet100':
            
            tidx0 = torch.tensor(train_dataset.targets) == args.task_id*10
            tidx1 = torch.tensor(train_dataset.targets) == args.task_id*10 + 1
            tidx2 = torch.tensor(train_dataset.targets) == args.task_id*10 + 2
            tidx3 = torch.tensor(train_dataset.targets) == args.task_id*10 + 3
            tidx4 = torch.tensor(train_dataset.targets) == args.task_id*10 + 4
            tidx5 = torch.tensor(train_dataset.targets) == args.task_id*10 + 5
            tidx6 = torch.tensor(train_dataset.targets) == args.task_id*10 + 6
            tidx7 = torch.tensor(train_dataset.targets) == args.task_id*10 + 7
            tidx8 = torch.tensor(train_dataset.targets) == args.task_id*10 + 8
            tidx9 = torch.tensor(train_dataset.targets) == args.task_id*10 + 9 

            vidx0 = torch.tensor(val_dataset.targets) == args.task_id*10
            vidx1 = torch.tensor(val_dataset.targets) == args.task_id*10 + 1
            vidx2 = torch.tensor(val_dataset.targets) == args.task_id*10 + 2
            vidx3 = torch.tensor(val_dataset.targets) == args.task_id*10 + 3
            vidx4 = torch.tensor(val_dataset.targets) == args.task_id*10 + 4
            vidx5 = torch.tensor(val_dataset.targets) == args.task_id*10 + 5
            vidx6 = torch.tensor(val_dataset.targets) == args.task_id*10 + 6
            vidx7 = torch.tensor(val_dataset.targets) == args.task_id*10 + 7
            vidx8 = torch.tensor(val_dataset.targets) == args.task_id*10 + 8
            vidx9 = torch.tensor(val_dataset.targets) == args.task_id*10 + 9

            train_mask = tidx0 | tidx1 | tidx2 | tidx3 | tidx4 | tidx5 | tidx6 | tidx7 | tidx8 | tidx9 
            train_indices = train_mask.nonzero().reshape(-1)

            val_mask = vidx0 | vidx1 | vidx2 | vidx3 | vidx4 | vidx5 | vidx6 | vidx7 | vidx8 | vidx9
            val_indices = val_mask.nonzero().reshape(-1)

            from torch.utils.data import Subset
            train_subset = Subset(train_dataset, train_indices)
            train_loader = DataLoader(train_subset, shuffle=True, batch_size=args.batch_size)

            val_subset = Subset(val_dataset, val_indices)
            val_loader = DataLoader(val_subset, shuffle=True, batch_size=args.batch_size)

        if args.dataset == 'TinyImagenet':

            tidx0 = torch.tensor(train_dataset.targets) == args.task_id*5
            tidx1 = torch.tensor(train_dataset.targets) == args.task_id*5 + 1
            tidx2 = torch.tensor(train_dataset.targets) == args.task_id*5 + 2
            tidx3 = torch.tensor(train_dataset.targets) == args.task_id*5 + 3
            tidx4 = torch.tensor(train_dataset.targets) == args.task_id*5 + 4
    
            vidx0 = torch.tensor(val_dataset.targets) == args.task_id*5
            vidx1 = torch.tensor(val_dataset.targets) == args.task_id*5 + 1
            vidx2 = torch.tensor(val_dataset.targets) == args.task_id*5 + 2
            vidx3 = torch.tensor(val_dataset.targets) == args.task_id*5 + 3
            vidx4 = torch.tensor(val_dataset.targets) == args.task_id*5 + 4
    
            train_mask = tidx0 | tidx1 | tidx2 | tidx3 | tidx4
            train_indices = train_mask.nonzero().reshape(-1)

            val_mask = vidx0 | vidx1 | vidx2 | vidx3 | vidx4
            val_indices = val_mask.nonzero().reshape(-1)

            from torch.utils.data import Subset
            train_subset = Subset(train_dataset, train_indices)
            train_loader = DataLoader(train_subset, shuffle=True, batch_size=args.batch_size)

            val_subset = Subset(val_dataset, val_indices)
            val_loader = DataLoader(val_subset, shuffle=True, batch_size=args.batch_size)

        if args.dataset == 'CIFAR100' or args.dataset == 'permMNIST' or args.dataset == 'splitMNIST':
            train_ = train_datasets[args.task_id]
            test_ = test_datasets[args.task_id]
            train_loader = DataLoader(train_, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(test_, batch_size=args.batch_size, shuffle=False)

        if args.dataset == '5_datasets':
            train_loader = trainloaders[args.task-id]
            val_loader = valloaders[args.task-id]

        if args.prune_method == 'lottery_ticket':  
            layers = {}
            percentage = {} 
            counter = 0
            state_dict = model.state_dict()

            for module_idx, module in enumerate(model.modules()):
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    percentage[module_idx] = args.V_min + random.random()*(args.V_max-args.V_min)
                    layers[module_idx] = counter
                
                    counter += 1

            mask_t = init_mask(model, layers, percentage, previous_masks, args.task_id, capacity, bit_width)
            manager = Manager(args, model, previous_masks, dataset2idx, dataset2biases, masks=mask_t, capacity=capacity, bit_width=bit_width)
  
            params_to_optimize = manager.model.parameters()
            optimizer = optim.Adam(params_to_optimize, lr=args.lr, betas=(0.9,0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)

        if args.prune_method == 'lottery_ticket_search':

            sparsities = []
            a_sparsities = []
            results = []
            size = compute_size(model)
            sz_before = compute_average_sparsity(model, previous_masks)

            fitness = []
            masks_list = []

            for solution_id in range(args.population_size):
                layers = {}
                percentage = {} 
                counter = 0
                state_dict = model.state_dict()

                for module_idx, module in enumerate(model.modules()):
                    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                        percentage[module_idx] = args.V_min + random.random()*(args.V_max-args.V_min)
                        layers[module_idx] = counter
                
                        counter += 1

                mask_t = init_mask(model, layers, percentage, previous_masks, args.task_id, capacity, bit_width)

                model_c = copy.deepcopy(model)
                manager = Manager(args, model_c, previous_masks, dataset2idx, dataset2biases, masks=mask_t, capacity=capacity, bit_width=bit_width)
                manager.pruner.current_masks = copy.deepcopy(previous_masks)
                manager.pruner.make_weights_random()
            
                params_to_optimize = manager.model.parameters()
                optimizer = optim.Adam(params_to_optimize, lr=args.lr, betas=(0.9,0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)

                manager.train(train_loader, val_loader, args.search_epochs, optimizer, save=True, directory=args.save_prefix, filename=args.checkpoint)
                errors = manager.eval(val_loader, manager.pruner.current_dataset_idx)

                sz = compute_average_sparsity(model_c, mask_t)
                results.append(100 - errors[0])
                sparsities.append(percentage)
                a_sparsities.append(1.0-(sz+sz_before)/size)
                fitness.append([0.8*(100.0-errors[0])/100.0+0.2*(1.0-(sz+sz_before)/size)])
                            
            #choose best
            id_ = np.argmax(fitness)
 
            mask_t = init_mask(model, layers, sparsities[id_], previous_masks, args.task_id, capacity, bit_width)
            manager = Manager(args, model, previous_masks, dataset2idx, dataset2biases, masks=mask_t, capacity=capacity, bit_width=bit_width)

            params_to_optimize = manager.model.parameters()
            optimizer = optim.Adam(params_to_optimize, lr=args.lr, betas=(0.9,0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
      
        if args.prune_method == 'dynamic_mask':
            
            #manager = Manager(args, model, previous_masks, dataset2idx, dataset2biases, prune_perc_per_layer=sparsities[id_], masks=None, capacity=capacity, bit_width=bit_width) 
            for module_idx, module in enumerate(model.modules()):
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    prune_perc_per_layer[module_idx] = args.V_min + random.random()*(args.V_max-args.V_min)
                    layers[module_idx] = counter
                
                    counter += 1

            model_c = copy.deepcopy(model)
            manager = Manager(args, model_c, previous_masks, dataset2idx, dataset2biases, prune_perc_per_layer=prune_perc_per_layer, masks=None, capacity=capacity, bit_width=bit_width)
            manager.pruner.current_masks = copy.deepcopy(previous_masks)
            manager.pruner.make_weights_random()
            
            params_to_optimize = manager.model.parameters()
            optimizer = optim.Adam(params_to_optimize, lr=args.lr, betas=(0.9,0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
 
        if args.prune_method == 'dynamic_mask_search':

            sparsities = []
            a_sparsities = []
            results = []
            size = compute_size(model)
            fitness = []
            #import pdb
            #pdb.set_trace()
            prune_perc_per_layer = {}

            for solution_id in range(args.population_size):
                layers = {}
                percentage = {} 
                counter = 0
                state_dict = model.state_dict()

                for module_idx, module in enumerate(model.modules()):
                    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                        prune_perc_per_layer[module_idx] = args.V_min + random.random()*(args.V_max-args.V_min)
                        layers[module_idx] = counter
                
                        counter += 1

                #mask_t = init_mask(model, layers, percentage, previous_masks, args.task_id)

                model_c = copy.deepcopy(model)
                manager = Manager(args, model_c, previous_masks, dataset2idx, dataset2biases, prune_perc_per_layer=prune_perc_per_layer, masks=None, capacity=capacity, bit_width=bit_width)
                manager.pruner.current_masks = copy.deepcopy(previous_masks)
                manager.pruner.make_weights_random()
            
                params_to_optimize = manager.model.parameters()
                optimizer = optim.Adam(params_to_optimize, lr=args.lr, betas=(0.9,0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)

                manager.train(train_loader, val_loader, args.search_epochs, optimizer, save=True, directory=args.save_prefix, filename=args.checkpoint)
                errors = manager.eval(val_loader, manager.pruner.current_dataset_idx)

                sz = compute_average_sparsity_by_percentage(model_c, prune_perc_per_layer)
                results.append(100 - errors[0])
                sparsities.append(percentage)
                a_sparsities.append(sz/size)
                fitness.append([0.8*(100.0-errors[0])/100.0+0.2*sz/size])
                            
            #import pdb
            #pdb.set_trace()
            #choose best
            id_ = np.argmax(fitness)
 
            #mask_t = init_mask(model, layers, sparsities, previous_masks, args.task_id)
            prune_perc_per_layer=sparsities[id_]
            manager = Manager(args, model, previous_masks, dataset2idx, dataset2biases, prune_perc_per_layer=sparsities[id_], masks=None, capacity=capacity, bit_width=bit_width)
          
            params_to_optimize = manager.model.parameters()
            optimizer = optim.Adam(params_to_optimize, lr=args.lr, betas=(0.9,0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)

        if args.mode == 'quantize_and_prune':
        
            #if args.prune_method == 'lottery_ticket':
            #    for module_idx, module in enumerate(model.modules()):
            #        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            #            manager.pruner.prune_perc_[module_idx] = 0.68        

            manager.pruner.current_masks = copy.deepcopy(previous_masks)
            ##manager.pruner.make_weights_random()

            params_to_optimize = manager.model.parameters()
            optimizer = optim.Adam(params_to_optimize, lr=args.lr, betas=(0.9,0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)

            manager.train(train_loader, val_loader, args.epochs, optimizer, save=True, directory=args.save_prefix, filename=args.checkpoint)
            errors = manager.eval(val_loader, manager.pruner.current_dataset_idx)     
        
            #manager = Manager(args, model, previous_masks, dataset2idx, dataset2biases, masks, capacity=capacity, bit_width=bit_width)
            manager.pruner.current_masks = copy.deepcopy(previous_masks)

            #manager.train(train_loader, val_loader, 8, optimizer, save=True, directory=args.save_prefix, filename=args.checkpoint)
            errors = manager.eval(val_loader, manager.pruner.current_dataset_idx)
            
            errors_ = errors
            clusters = 64
            last = {}
            
            while abs((100 - errors_[0]) - (100 - errors[0])) < 2.0:

                #init capacity
                model_ = copy.deepcopy(model)

                if args.mode == 'lottery_ticket' or args.mode == 'lottery_ticket_search':
                    manager = Manager(args, model_, previous_masks, dataset2idx, dataset2biases, masks=mask_t, capacity=capacity, bit_width=bit_width)
                    manager.pruner.current_masks = copy.deepcopy(previous_masks)

                if args.mode == 'dynamic_mask' or args.mode == 'dynamic_mask_search':
                    manager = Manager(args, model_, previous_masks, dataset2idx, dataset2biases, prune_perc_per_layer=prune_perc_per_layer, masks=None, capacity=capacity, bit_width=bit_width)
                    manager.pruner.current_masks = copy.deepcopy(previous_masks)                

                for module_idx, module in enumerate(manager.model.modules()):
                    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                        new_weight = module.weight.data*(manager.pruner.current_masks[module_idx] == manager.pruner.current_dataset_idx).float().cuda()
                        other_weights = module.weight.data*(manager.pruner.current_masks[module_idx] != manager.pruner.current_dataset_idx).float().cuda()

                        q_weight, values, labels = vquant(new_weight, n_clusters=clusters)                 
                        q_weight = torch.from_numpy(q_weight).cuda()                
                
                        q_weight[manager.pruner.current_masks[module_idx] != manager.pruner.current_dataset_idx] = 0
                        labels[manager.pruner.current_masks[module_idx].cpu().numpy() != manager.pruner.current_dataset_idx] = -1

                        new_weight = q_weight*(manager.pruner.current_masks[module_idx] == (manager.pruner.current_dataset_idx)).float().cuda()
                        module.weight.data = new_weight + other_weights
                        
                errors_ = manager.eval(val_loader, manager.pruner.current_dataset_idx)

                if abs((100 - errors_[0]) - (100 - errors[0])) < 2.0:
                    for module_idx, module in enumerate(manager.model.modules()):
                        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                            last[module_idx] = module.weight.data 

                print(clusters)
                clusters = clusters // 2

                #del manager

            if args.mode == 'lottery_ticket' or args.mode == 'lottery_ticket_search':
                manager = Manager(args, model, previous_masks, dataset2idx, dataset2biases, masks=mask_t, capacity=capacity, bit_width=bit_width)
                manager.pruner.current_masks = copy.deepcopy(previous_masks)

            if args.mode == 'dynamic_mask' or args.mode == 'dynamic_mask_search':
                manager = Manager(args, model, previous_masks, dataset2idx, dataset2biases, prune_perc_per_layer=prune_perc_per_layer, masks=None, capacity=capacity, bit_width=bit_width)
                manager.pruner.current_masks = copy.deepcopy(previous_masks)

            import pdb
            pdb.set_trace()
            #set best quant
            for module_idx, module in enumerate(manager.model.modules()):
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    #new_weight = module.weight.data*(manager.pruner.current_masks[module_idx] == manager.pruner.current_dataset_idx).float().cuda()
                    #other_weights = module.weight.data*(manager.pruner.current_masks[module_idx] != manager.pruner.current_dataset_idx).float().cuda()

                    #q_weight, values, labels = vquant(new_weight, n_clusters=clusters*4)                 
                    #q_weight = torch.from_numpy(q_weight).cuda()                
                
                    #q_weight[manager.pruner.current_masks[module_idx] != manager.pruner.current_dataset_idx] = 0
                    #labels[manager.pruner.current_masks[module_idx].cpu().numpy() != manager.pruner.current_dataset_idx] = -1

                    #new_weight = q_weight*(manager.pruner.current_masks[module_idx] == (manager.pruner.current_dataset_idx)).float().cuda()
                    module.weight.data = last[module_idx]

            errors = manager.eval(val_loader, manager.pruner.current_dataset_idx)
            manager.pruner.make_finetuning_mask(bit_width=math.log(clusters*4, 2))     
            #errors = manager.eval(val_loader, manager.pruner.current_dataset_idx-1)
            #get batch norms and save
            #get batch norms
            
            batchnorms = get_batchnorms(manager.model)
            import pdb
            pdb.set_trace()
            manager.save_model(args.epochs, 0.0, errors, directory=args.save_prefix, filename=args.checkpoint, capacity=capacity, batch_norms=batchnorms)

            model_bp = copy.deepcopy(manager.model)
     
        elif args.mode == 'eval':     
            # Just run the model on the eval set.
            manager.eval(manager.pruner.current_dataset_idx-1)

if __name__ == '__main__':
    main()