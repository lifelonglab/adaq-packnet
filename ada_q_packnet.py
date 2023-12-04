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
import cl_networks as net
import utils as utils
from prune import SparsePruner
from load import get_multitask_experiment
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
FLAGS.add_argument('--lr', type=float, default=0.01,
                   help='Learning rate')
FLAGS.add_argument('--lr_decay_every', type=int,
                   help='Step decay every this many epochs')
FLAGS.add_argument('--lr_decay_factor', type=float,
                   help='Multiply lr by this much every step of decay')
FLAGS.add_argument('--epochs', type=int, default=10,
                   help='Number of initial finetuning epochs')
FLAGS.add_argument('--search_epochs', type=int, default=10,
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
FLAGS.add_argument('--save_prefix', type=str, default='/work/packnet/checkpoints/',
                   help='Location to save model')
FLAGS.add_argument('--loadname', type=str, default='cifar100_0.pt',
                   help='Location to save model')
FLAGS.add_argument('--initname', type=str, default='cifar100_0.pt',
                   help='Location to init model')

# Pruning options.
FLAGS.add_argument('--prune_method', type=str,
                   choices=['lottery_ticket', 'lottery_ticket_search', 'fine_tuning', 'dynamic_mask', 'dynamic_mask_search'],
                   help='Pruning method to use')
FLAGS.add_argument('--prune_perc_per_layer', type=float, default=0.6,
                   help='% of neurons to prune per layer')
FLAGS.add_argument('--disable_pruning_mask', action='store_true', default=False,
                   help='use masking or not')
FLAGS.add_argument('--train_biases', action='store_true', default=False,
                   help='use separate biases or not')
FLAGS.add_argument('--train_bn', action='store_true', default=False,
                   help='train batch norm or not')
# Other.
FLAGS.add_argument('--quantization_method', type=str,
                   choices=['nonlinear'],
                   help='Pruning method to use')
FLAGS.add_argument('--cuda', action='store_true', default=False,
                   help='use CUDA')
FLAGS.add_argument('--init_dump', action='store_true', default=False,
                   help='Initial model dump.')
FLAGS.add_argument('--task_number', type=int, default=10)
FLAGS.add_argument('--task_id', type=int, default=0)
FLAGS.add_argument('--V_min', type=float, default=0.4)
FLAGS.add_argument('--V_max', type=float, default=0.4)
    

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


def init_mask(model, layers, percentage, previous_masks, task_id):
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
                    idxs = (previous_masks[counter].flatten() == task_id+1).nonzero()
                    idxs = idxs.numpy()

                    mask_[counter] = np.ones((sz[0], sz[1], sz[2], sz[3]))

                    #elem = np.random.choice(mask_[counter].size, int(percentage[counter]*mask_[counter].size), replace=False)
                    elem = np.random.choice(idxs.size, int(percentage[counter]*idxs.size), replace=False)
                    mask_[counter] = mask_[counter].flatten()
                    mask_[counter][idxs[elem]] = 0.0

                    mask_[counter] = np.reshape(mask_[counter], (sz[0], sz[1], sz[2], sz[3]))

            if (len(sz) == 2):
                    idxs = (previous_masks[counter].flatten() == task_id+1).nonzero()
                    idxs = idxs.numpy()

                    mask_[counter] = np.ones((sz[0], sz[1]))

                    #elem = np.random.choice(mask_[counter].size, int(percentage[counter]*mask_[counter].size), replace=False)
                    elem = np.random.choice(idxs.size, int(percentage[counter]*idxs.size), replace=False)
                    mask_[counter] = mask_[counter].flatten()
                    mask_[counter][idxs[elem]] = 0.0
                    mask_[counter] = np.reshape(mask_[counter], (sz[0], sz[1]))
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

    for module_idx, module in enumerate(model.modules()):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            mask = torch.ByteTensor(module.weight.data.size()).fill_(1)
            if 'cuda' in module.weight.data.type():
                mask = mask.cuda()
            previous_masks[module_idx] = mask
            

    
    torch.save({
        'dataset2idx': 1,
        'previous_masks': previous_masks,
        'model': model,
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
   
    previous_masks = ckpt['previous_masks']
    dataset2idx = ckpt['dataset2idx']
    #masks = ckpt['masks']
    if 'dataset2biases' in ckpt:
        dataset2biases = ckpt['dataset2biases']
    else:
        dataset2biases = {}
   
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
            mask_t = init_mask(model, layers, percentage, previous_masks, args.task_id)

        if args.prune_method == 'lottery_ticket_search':
            layers = {}
            percentage = {} 
            counter = 0
            state_dict = model.state_dict()

            for module_idx, module in enumerate(model.modules()):
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    percentage[module_idx] = args.V_min + random.random()*(args.V_max-args.V_min)
                    layers[module_idx] = counter
                
                    counter += 1
            mask_t = init_mask(model, layers, percentage, previous_masks, args.task_id)

            model_c = copy.deepcopy(model)
            manager = Manager(args, model_c, previous_masks, dataset2idx, dataset2biases, masks=mask_t)
            manager.pruner.current_masks = copy.deepcopy(previous_masks)
            ##manager.pruner.make_weights_random()
            
            params_to_optimize = manager.model.parameters()
            optimizer = optim.Adam(params_to_optimize, lr=args.lr, betas=(0.9,0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)

            manager.train(train_loader, val_loader, args.search_epochs, optimizer, save=True, directory=args.save_prefix, filename=args.checkpoint)
            errors = manager.eval(val_loader, manager.pruner.current_dataset_idx)
            

        import pdb
        pdb.set_trace()

        manager = Manager(args, model, previous_masks, dataset2idx, dataset2biases, mask_t)

        if args.mode == 'quantize_and_prune':
        
            #if args.prune_method == 'lottery_ticket':
            #    for module_idx, module in enumerate(model.modules()):
            #        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            #            manager.pruner.prune_perc_[module_idx] = 0.68        

            manager.pruner.current_masks = copy.deepcopy(previous_masks)
            ##manager.pruner.make_weights_random()
            

            params_to_optimize = manager.model.parameters()
            optimizer = optim.Adam(params_to_optimize, lr=args.lr, betas=(0.9,0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)

            manager.train(train_loader, val_loader, args.epochs, optimizer, save=True, savename=args.save_prefix, prune=True)
            errors = manager.eval(val_loader, manager.pruner.current_dataset_idx)           
        
            manager = Manager(args, model, previous_masks, dataset2idx, dataset2biases, masks)
            manager.pruner.current_masks = copy.deepcopy(previous_masks)

            #manager.train(train_loader, val_loader, 8, optimizer, save=True, savename=args.save_prefix, prune=True)
            errors = manager.eval(val_loader, manager.pruner.current_dataset_idx)
            
            errors_ = errors
            clusters = 64
            
            while (100 - errors_[0]) - (100 - errors[0]):
                model_ = copy.deepcopy(model)
                manager = Manager(args, model_, previous_masks, dataset2idx, dataset2biases, train_, test_, masks)
                for module_idx, module in enumerate(manager.model.modules()):
                    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                        new_weight = module.weight.data*(manager.pruner.current_masks[module_idx] == manager.pruner.current_dataset_idx).float()
                        other_weights = module.weight.data*(manager.pruner.current_masks[module_idx] != manager.pruner.current_dataset_idx).float()

                        q_weight, values, labels = vquant(new_weight, n_clusters=clusters)                 
                        q_weight = torch.from_numpy(q_weight).cuda()                
                
                        q_weight[manager.pruner.current_masks[module_idx] != manager.pruner.current_dataset_idx] = 0
                        labels[manager.pruner.current_masks[module_idx].cpu().numpy() != manager.pruner.current_dataset_idx] = -1

                        new_weight = q_weight*(manager.pruner.current_masks[module_idx] == (manager.pruner.current_dataset_idx)).float()
                        module.weight.data = new_weight + other_weights

                clusters = clusters // 2 
                errors_ = manager.eval(val_loader, manager.pruner.current_dataset_idx)
                del manager

            #set best quant
            import pdb
            pdb.set_trace()

            manager.pruner.make_finetuning_mask()     
            errors = manager.eval(manager.pruner.current_dataset_idx-1)
            model_bp = copy.deepcopy(manager.model)
     
        elif args.mode == 'eval':     
            # Just run the model on the eval set.
            manager.eval(manager.pruner.current_dataset_idx-1)

if __name__ == '__main__':
    main()