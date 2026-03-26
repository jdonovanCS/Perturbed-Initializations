# from distutils.command.config import config
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import randomdatamodule as rd
import matplotlib.pyplot as plt
import scikits.bootstrap as bootstrap
import warnings
warnings.filterwarnings('ignore')
import wandb
import pl_bolts.datamodules
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from net import Net
from big_net import Net as BigNet
from ae_net import AE
from cifar100datamodule import CIFAR100DataModule
from tinyimagenetdatamodule import TinyImageNetDataModule
import numba
import os
import random
import collections
import _collections_abc
import collections.abc
import gc
from vgg16 import Net as vgg16
from v_net import Net as vNet
import torch.nn.functional as F

# from pl_bolts.utils.self_supervised import UnderReviewWarning


def create_random_images(num_images=200):
    paths = []
    for i in range(num_images):
        if not os.path.exists('images/random/{}.png'.format(i)):
            rgb = np.random.randint(255, size=(32,32,3), dtype=np.uint8)
            cv2.imwrite('images/random/{}.png'.format(i), rgb)
        paths.append('images/random/{}.png'.format(i))
    return paths

# def load_random_images(random_image_paths, batch_size=64):
#     train_dataset = rd.RandomDataset(random_image_paths)
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
#     return train_loader
from pytorch_lightning.plugins import DDPPlugin
def train_network(data_module, filters=None, epochs=2, lr=.001, save_path=None, fixed_conv=False, val_interval=1, novelty_interval=None, diversity={'type':'absolute', 'pdop':None, 'ldop':None, 'k': None, 'k_strat':True}, scaled=False, devices=1, save_interval=None, bn=True, log_activations=False):
    # check which dataset and get the classes for it
    gc.collect()
    torch.cuda.empty_cache()
    if data_module.num_classes < 101:
        classnames = list(data_module.dataset_test.classes)
    else:
        # classnames = list([x[0] for x in data_module.train_dataloader().dataset.classes])
        try:
            classnames = data_module.dataset_train.get_classes()
        except:
            classnames = None
    # check which network and instantiate it
    if scaled:
        total_devices = torch.cuda.device_count()
        device = torch.device(glob_rank % total_devices if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        # torch.distributed.init_process_group(backend='gloo',
        #                              init_method='env://')
        print(device)
        net = vgg16(num_classes=data_module.num_classes, classnames=classnames, diversity=None, lr=lr, bn=bn, log_activations=log_activations)
        net = net.to(device)
    elif len(filters) == 6:
        net = Net(num_classes=data_module.num_classes, classnames=classnames, diversity=diversity, lr=lr, bn=bn, data_dims=data_module.dims, log_activations=log_activations)
        # device = torch.device(0)
        # net = net.to(device)
    else:
        net = vNet(num_classes=data_module.num_classes, classnames=classnames, diversity=diversity, lr=lr, size=len(filters))
    print(net.device)
    if filters is not None:
        if not scaled:
            for i in range(len(net.conv_layers)):
                if i < len(filters):
                    z = torch.tensor(filters[i])
                    z = z.type_as(net.conv_layers[i].weight.data)
                    # z.to(net.device)
                    net.conv_layers[i].weight.data = z
                    # print(net.conv_layers[i].weight.data == filters[i].to(device))
                if fixed_conv:
                    for param in net.conv_layers[i].parameters():
                        param.requires_grad = False
                    for param in net.BatchNorm1.parameters():
                        param.requires_grad = False
                    for param in net.BatchNorm2.parameters():
                        param.requires_grad = False
                    for param in net.BatchNorm3.parameters():
                        param.requires_grad = False
        elif scaled:
            count = 0
            for m in net.model.modules():
                if isinstance(m, (torch.nn.Conv2d)):
                    z = torch.tensor(filters[count])
                    z = z.type_as(m.weight.data)
                    m.weight.data = z
                    count += 1
                if fixed_conv:
                    if isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
                        for param in m.parameters():
                            param.requires_grad=False

    if save_path is None:
        save_path = PATH
    callbacks=None
    if save_interval is not None:
        callbacks=[pl.callbacks.ModelCheckpoint(every_n_epochs=save_interval,save_top_k=-1)]

    if not scaled:
        wandb_logger = WandbLogger(log_model=True, log_graph=False)
    else:
        wandb_logger = WandbLogger(log_model=True, log_graph=False)
    print((torch.cuda.device_count()))
    accelerator = "cpu" if torch.cuda.device_count() < 1 else 'gpu'
    # trainer = pl.Trainer(max_epochs=epochs, default_root_dir=save_path, logger=wandb_logger, check_val_every_n_epoch=val_interval, accelerator="gpu", gpus=torch.cuda.device_count(), strategy='dp')
    if torch.cuda.device_count() > 1:
        trainer = pl.Trainer(callbacks=callbacks,max_epochs=epochs, default_root_dir=save_path, logger=wandb_logger, check_val_every_n_epoch=val_interval, accelerator=accelerator, devices=devices, plugins=DDPPlugin(find_unused_parameters=False))
    else:
        trainer = pl.Trainer(callbacks=callbacks, max_epochs=epochs, default_root_dir=save_path, logger=wandb_logger, check_val_every_n_epoch=val_interval, accelerator=accelerator)
    wandb_logger.watch(net, log_graph=False)
    # find_best_lr(trainer, net, data_module)
    # torch.cuda.empty_cache()
    trainer.fit(net, datamodule=data_module)

    # torch.save(net.state_dict(), save_path)
    # return record_progress

def train_vgg16(data_module, epochs=2, lr=.001, val_interval=4):
    # check which dataset and get the classes for it
    if data_module.num_classes < 101:
        classnames = list(data_module.dataset_test.classes)
    else:
        classnames = list([x[0] for x in data_module.train_dataloader().dataset.classes])

    net = vgg16(num_classes=data_module.num_classes, classnames=classnames, diversity=None, lr=lr)
    
    wandb_logger = WandbLogger(log_model=True)
    trainer = pl.Trainer(max_epochs=epochs, logger=wandb_logger, check_val_every_n_epoch=val_interval, accelerator="gpu")#, devices=1, plugins=DDPPlugin(find_unused_parameters=False))

    wandb_logger.watch(net, log="all")
    trainer.fit(net, datamodule=data_module)

def train_ae_network(data_module, epochs=100, steps=-1, lr=.001, encoded_space_dims=256, save_path=None, novelty_interval=4, val_interval=1, diversity={'type':'absolute', 'pdop':None, 'ldop':None, 'k': None, 'k_strat':True}, scaled=False, rand_tech='uniform'):
    net = AE(encoded_space_dims, diversity, lr)
    if rand_tech == 'normal':
        normalize(net)
    # net = net.to(device)
    if save_path is None:
        save_path = PATH
    wandb_logger = WandbLogger(log_model=True)
    trainer = pl.Trainer(max_epochs=epochs, max_steps=steps, default_root_dir=save_path, logger=wandb_logger, check_val_every_n_epoch=val_interval, accelerator='gpu', gpus=torch.cuda.device_count(), strategy='dp')
    wandb_logger.watch(net, log='all')
    trainer.fit(net, datamodule=data_module)

def get_data_module(dataset, batch_size, workers=np.inf, shuffle=False):
    match dataset.lower():
        case 'cifar10' | 'cifar-10':
            data_module = pl_bolts.datamodules.CIFAR10DataModule(batch_size=batch_size, data_dir="data/", num_workers=min(workers, os.cpu_count()), pin_memory=True, shuffle=shuffle)
        case 'cifar100' | 'cifar-100':
            data_module = CIFAR100DataModule(batch_size=batch_size, data_dir="data/", num_workers=min(workers, os.cpu_count()), pin_memory=True)
        case 'imagenet':
            data_module = pl_bolts.datamodules.ImagenetDataModule(batch_size=batch_size, data_dir="data/imagenet/", num_workers=min(workers, os.cpu_count()), pin_memory=True)
        case 'miniimagenet':
            data_module = pl_bolts.datamodules.ImagenetDataModule(batch_size=batch_size, data_dir="data/miniimagenet/", num_workers=min(workers, os.cpu_count()), pin_memory=True, shuffle=shuffle)
        case'tinyimagenet':
            data_module = TinyImageNetDataModule(batch_size=batch_size, data_dir="data/tinyimagenet", num_workers=min(workers, os.cpu_count()), pin_memory=True)
        case 'random':
            data_module = rd.RandomDataModule(data_dir='images/random/', batch_size=batch_size, num_workers=min(workers, os.cpu_count()), pin_memory=True)
        case _:
            print('Please supply dataset of CIFAR-10 or CIFAR-100')
            exit()
    return data_module

def save_npy(filename, data, index=0):
    if not os.path.isfile(filename) or index==0:
        with open(filename, 'wb') as f:
            np.save(f, data)
    else:
        with open(filename, 'rb') as f:
            before = np.load(f, allow_pickle=True)
        with open(filename, 'wb') as f:
            after = np.append(before, [data[index]], axis=0)
            np.save(f, after)
    

@numba.njit(parallel=True)
def diversity(acts, pdop=None, k=-1, k_strat=None):

    I = len(acts)
    C = len(acts[0])
    
    # make array correct size
    if k_strat == 'random':
        pairwise = np.zeros((I,C,k))
    else:
        pairwise = np.zeros((I,C,C))

    for image in numba.prange(I):
        for channel in range(C):
            # if strategy is random, then we do fewer calculations overall
            if k_strat=='random':
                choices = np.random.choice(C, k, replace=False)
                for i in range(k):
                    div = np.abs(acts[image, channel] - acts[image, choices[i]]).sum()
                    pairwise[image, channel, i] = div
            # otherwise we have to loop for every other channel in the layer
            else:
                for channel2 in range(channel, C):
                    div = np.abs(acts[image, channel] - acts[image, channel2]).sum()
                    pairwise[image, channel, channel2] = div
                    pairwise[image, channel2, channel] = div
    
    # if we are using k-neighbors, then we want a subset of the calculated results above
    if k > 0 and k_strat != 'random':
        for image in range(I):
            for channel in range(C):
                pairwise[image, channel] = sorted(pairwise[image, channel], reverse=k_strat=='furthest')[0:k]

    # set a new variable up as a copy of pairwise. Numba fails without this.
    res = pairwise

    if pdop == 'sum':
        return((res).sum())
    elif pdop == 'mean':
        return((res).mean())
    elif pdop == 'rms':
        return(np.sqrt(np.mean(res**2)))

@numba.njit(parallel=True)
def diversity_orig(acts, pdop="", k=0, k_strat=""):
    B = len(acts)
    I = len(acts[0])
    if k_strat == 'random':
        pairwise = np.zeros((B, I, k))
    else:
        pairwise = np.zeros((B, I, I))

    for batch in range(B):
        for image in numba.prange(I):
            if k_strat=='random':
                choices = np.random.choice(I, k, replace=False)
                for i in range(k):
                    div = np.abs(acts[batch][choices[i]] - acts[batch][image]).sum()
                    pairwise[batch, image, i] = div
            else:
                for image2 in range(I):
                    div = np.abs(acts[batch][image2] - acts[batch][image]).sum()
                    pairwise[batch, image, image2] = div
                    pairwise[batch, image2, image] = div
                
    if k > 0 and k_strat != 'random':
        for batch in range(B):
            for image in range(I):
                pairwise[batch, image] = sorted(pairwise[batch, image], reverse=k_strat=='furthest')[0:k]

    res = pairwise
                   
    if pdop == 'sum':
        return((res).sum())
    elif pdop == 'mean':
        return((res).mean())
    elif pdop == 'rms':
        return(np.sqrt(np.mean(res**2)))

@numba.njit(parallel=True)
def diversity_relative(acts, pdop="", k=0, k_strat=""):
    I=len(acts)
    C=len(acts[0])
    
    if k_strat=='random':
        pairwise = np.zeros((I,C,k))
    else:
        pairwise = np.zeros((I,C,C))
    
    for image in numba.prange(I):
        for channel in range(C):
            if k_strat == 'random':
                choices = np.random.choice(C, k, replace=False)
                for i in range(k):
                    dist = np.abs(acts[image, channel]-acts[image, choices[i]]).sum()
                    divisor = (np.abs(acts[image, channel]).sum()) + (np.abs(acts[image, choices[i]]).sum())
                    div=(dist / divisor)
                    pairwise[image, channel, i] = div
            else:
                for channel2 in range(channel+1, C):
                    dist = np.abs(acts[image, channel]-acts[image, channel2]).sum()
                    divisor = (np.abs(acts[image, channel]).sum()) + (np.abs(acts[image, channel2]).sum())
                    div=(dist / divisor)
                    # div=np.abs((acts[batch, channel]-acts[batch, channel2])/(acts[batch, channel]+acts[batch, channel2])).sum()
                    pairwise[image, channel, channel2] = div
                    pairwise[image, channel2, channel] = div
    retVal = 0
    if k > 0 and k_strat != 'random':
        for image in range(I):
            for channel in range(C):
                if pdop == 'sum':
                    retVal += (np.array(sorted(pairwise[image, channel], reverse=k_strat=='furthest')[0:k]).sum())
                elif pdop == 'mean':
                    retVal += (1.0/(I*C))*(np.array(sorted(pairwise[image, channel], reverse=k_strat=='furthest')[0:k]).mean())
                elif pdop == 'rms':
                    retVal += (1.0/(I*C))*(np.sqrt(np.mean(np.array(sorted(pairwise[image, channel], reverse=k_strat=='furthest')[0:k])**2)))
        return retVal

    res = pairwise

    if pdop == 'sum':
        return((res).sum())
    elif pdop == 'mean':
        return((res).mean())
    elif pdop == 'rms':
        return(np.sqrt(np.mean(res**2)))

@numba.njit(parallel=True, fastmath=True)
def diversity_cosine_distance(acts, pdop="", k=0, k_strat=""):
    I=len(acts)
    C=len(acts[0])

    if k_strat == 'random':
        pairwise = np.zeros((I,C,k))
    else:
        pairwise = np.zeros((I,C,C))
    
    for image in numba.prange(I):
        for channel in range(C):
            c_flat = acts[image, channel].flatten()
            if k_strat == 'random':
                choices = np.random.choice(C, k, replace=False)
                for i in range(k):
                    c2_flat = acts[image, choices[i]].flatten()
                    dist = cosine_dist(c_flat, c2_flat)
                    pairwise[image, channel, i] = dist
            else:
                for channel2 in range(channel+1, C):
                    c2_flat = acts[image, channel2].flatten()
                    dist = cosine_dist(c_flat, c2_flat)
                    pairwise[image, channel, channel2] = dist
                    pairwise[image, channel2, channel] = dist
    if k > 0 and k_strat != 'random':
        for image in range(I):
            for channel in range(C):
                pairwise[image, channel] = sorted(pairwise[image, channel], reverse=k_strat=='furthest')[0:k]
    
    res = pairwise

    if pdop == 'sum':
        return((res).sum())
    elif pdop == 'mean':
        return((res).mean())
    elif pdop == 'rms':
        return(np.sqrt(np.mean(res**2)))

@numba.njit(parallel=True, fastmath=True)
def cosine_dist(u:np.ndarray, v:np.ndarray):
    uv=0
    uu=0
    vv=0
    for i in range(u.shape[0]):
        uv+=u[i]*v[i]
        uu+=u[i]*u[i]
        vv+=v[i]*v[i]
    cos_theta=1
    if uu!=0 and vv!=0:
        cos_theta=uv/np.sqrt(uu*vv)
    return 1-cos_theta

# @numba.mjit(parallel=True, fastmath=True)
def get_activation_covariance(activations):

    B, C, H, W = activations.shape

    # Reshape
    fm = activations.view(B, C, -1)

    cov_matrices = []
    for b in range(B):
        f = fm[b]
        cov = f@f.T / (f.shape[1]-1)
        cov_matrices.append(cov)

    return torch.stack(cov_matrices).mean(dim=0)

def get_activation_cosine_distance(activations):
    B, C, H, W = activations.shape
    fm = activations.view(B, C, -1)

    fm_norm = F.normalize(fm, p=2, dim=2)

    cosine_dist_matrices = []
    for b in range(B):
        f = fm_norm[b]
        sim = f @ f.T
        dist = 1 - sim
        cosine_dist_matrices.append(dist)
    
    return torch.stack(cosine_dist_matrices).mean(dim=0)

#TODO - orthonormalize filters to one another. Flipping dimensions may not be doing what we want. 
# Google how to detemine of two matrices are orthognal. dotproduct and crossproduct or outer product.
# Look at randomly initialized filters as well for this property.
@numba.njit(parallel=True)
def gram_shmidt_orthonormalize(filters):
# for layer_filters in filters:
    for f in range(len(filters)):
        copied = filters[f].copy()
        shape_0 = filters[f].shape[0]
        shape_1=np.prod(np.array(filters[f].shape[1:]))
        input = copied.reshape((max(shape_0, shape_1), min(shape_0, shape_1)))
        q,r = np.linalg.qr(input)
        copied = q.copy()
        f = copied.reshape(filters[f].shape)
    return filters

def feature_novelty(features, history):
    novelty = 0

    for feat in history:
        for i, l in enumerate(feat):
            novelty += np.sum(np.abs(features[i]-l))
    return novelty

def filter_novelty(filters, history):
    novelty = 0

    for filter in history:
        for i, l in enumerate(filter):
            novelty += np.sum(np.abs(filters[i]-l))
    return novelty

# @numba.njit(parallel=True)
def normalize(net):
    for m in net.modules():
        kaiming_normalize(m)
        
def kaiming_normalize(m):
    if getattr(m, 'bias', None) is not None:
        torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.kaiming_normal_(m.weight)
    for l in m.children(): kaiming_normalize(l)
# from tqdm import tqdm
# def gram_schmidt(vectors):
#     basis = []
#     for v in tqdm(range(len(vectors))):
#         w = vectors[v] - np.sum( np.dot(vectors[v],b)*b  for b in basis )
#         if (w > 1e-10).any():  
#             basis.append(w/np.linalg.norm(w))
#     return np.array(basis)
    
def xavier_uniform(net, gain):
    for m in net.modules():
        xavier_uniform_inner(m, gain)

def xavier_uniform_inner(m, gain):
    if getattr(m, 'bias', None) is not None:
        torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
    for l in m.children(): xavier_uniform_inner(l, gain)


def xavier_normal(net, gain):
    for m in net.modules():
        xavier_normal_inner(m, gain)

def xavier_normal_inner(m, gain):
    if getattr(m, 'bias', None) is not None:
        torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.xavier_normal_(m.weight, gain=gain)
    for l in m.children(): xavier_normal_inner(l, gain)

def orthogonal(net, gain):
    for m in net.modules():
        orthogonal_inner(m, gain)

def orthogonal_inner(m, gain):
    if getattr(m, 'bias', None) is not None:
        torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.orthogonal_(m.weight, gain=gain)
    for l in m.children(): orthogonal_inner(l, gain)

def default_uniform(net):
    for m in net.modules():
        default_uniform_inner(m)

def default_uniform_inner(m):
    if getattr(m, 'bias', None) is not None:
        torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.uniform_(m.weight)
    for l in m.children(): default_uniform_inner(l)

def default_normal(net):
    for m in net.modules():
        default_normal_inner(m)

def default_normal_inner(m):
    if getattr(m, 'bias', None) is not None:
        torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.normal_(m.weight)
    for l in m.children(): default_normal_inner(l)

def get_dist(layer_params):
    num_outside = 0
    for layer in range(len(layer_params)):
        pdf = mean = std = abs_mean = 0
        divisor = max(abs(layer_params[layer].flatten()))
        multiplier = max(abs(layer_params[layer].flatten()))
        num_bins_ = int(100*multiplier/divisor)
        num_outside = 0
        num_outside += sum(abs(layer_params[layer].flatten()) > divisor)

        # max(np.abs(layer_params[layer].flatten()))
        # getting data of the histogram
        count, bins_count = np.histogram(layer_params[layer].flatten(), bins=num_bins_, normed=True)
            
        # verify sum to 1
        widths = bins_count[1:] - bins_count[:-1]
        assert sum(count * widths) > .99 and sum(count * widths) < 1.01

        # finding the PDF of the histogram using count values
        pdf += count / sum(count)

        mean += layer_params[layer].flatten().mean()
        std += layer_params[layer].flatten().std()
        abs_mean += abs(layer_params[layer].flatten()).mean()
            
        # using numpy np.cumsum to calculate the CDF
        # We can also find using the PDF values by looping and adding
        # cdf = np.cumsum(pdf)
        # cdf_r = np.cumsum(pdf_r)
            
        pdf = pdf / len(layer_params)
        mean = mean / len(layer_params)
        std = std / len(layer_params)
        abs_mean = abs_mean / len(layer_params)

        return pdf, mean, std, abs_mean


@numba.njit(parallel=True)
def diversity_constant(acts):
    constants = np.zeros((len(acts)))
    for i in numba.prange(len(acts)):
        constants[i] = i
    return sum(constants)

def choose_mutate_index_from_layer(filters, layer):
    filters_in_layer = len(filters[layer].flatten())
    count = sum([len(x.flatten()) for e, x in enumerate(filters) if e < layer])
    rand_filter = random.randint(1, filters_in_layer) + count
    selected_dims = [0,0]
    
    for i in range(len(filters[layer])):
        if(len(filters[layer][i].flatten())) +count > rand_filter:
            selected_dims[0] = i
            break
        count += len(filters[layer][i].flatten())
    
    for i in range(len(filters[layer][selected_dims[0]])):
        if len(filters[layer][selected_dims[0]][i].flatten()) + count > rand_filter:
            selected_dims[1] = i
            break
        count += len(filters[layer][selected_dims[0]][i].flatten())
    
    return (layer, selected_dims[0], selected_dims[1])

def choose_mutate_index(filters, weighted_mut=False, weights_for_mut=None):
    if weighted_mut and not weights_for_mut:
        total_filters = sum([len(x.flatten()) for x in filters])
        rand_filter = random.randint(1, total_filters)
        selected_layer = 0
        selected_dims = [0,0]
        count = 0
        for i in range(len(filters)):
            if len(filters[i].flatten()) + count > rand_filter:
                selected_layer = i
                break
            count += len(filters[i].flatten())
        
        for i in range(len(filters[selected_layer])):
            if len(filters[selected_layer][i].flatten()) + count > rand_filter:
                selected_dims[0] = i
                break
            count += len(filters[selected_layer][i].flatten())
        
        for i in range(len(filters[selected_layer][selected_dims[0]])):
            if len(filters[selected_layer][selected_dims[0]][i].flatten()) + count > rand_filter:
                selected_dims[1] = i
                break
            count += len(filters[selected_layer][selected_dims[0]][i].flatten())
        # print(rand_filter)
        # print(selected_layer, selected_dims)
    elif weighted_mut and weights_for_mut:
        total_filters = int(sum([len(filters[i].flatten())*weights_for_mut[i] for i in range(len(filters))]))
        rand_layer = random.randint(0, total_filters-1)
        count = 0
        for i in range(len(weights_for_mut)):
            if weights_for_mut[i]*len(filters[i].flatten()) + count > rand_layer:
                selected_layer = i
                break
            count += weights_for_mut[i]*len(filters[i].flatten())

        rand_filter = random.randint(1, int(len(filters[selected_layer].flatten())))
        count = 0
        selected_dims = [0,0]

        for i in range(len(filters[selected_layer])):
            if len(filters[selected_layer][i].flatten()) + count > rand_filter:
                selected_dims[0] = i
                break
            count += len(filters[selected_layer][i].flatten())
        
        for i in range(len(filters[selected_layer][selected_dims[0]])):
            if len(filters[selected_layer][selected_dims[0]][i].flatten()) + count > rand_filter:
                selected_dims[1] = i
                break
            count += len(filters[selected_layer][selected_dims[0]][i].flatten())
       
    else:
        selected_layer = random.randint(0,len(filters)-1)
        selected_dims = []
        for v in list(filters[selected_layer].shape)[0:2]:
            selected_dims.append(random.randint(0,v-1))

    return (selected_layer, selected_dims[0], selected_dims[1])


def mutate(filters, broad_mutation=False, mr=1.0, dims=None):

    if not broad_mutation:
        # select a single 3x3 filter in one of the convolutional layers and replace it with a random new filter.
        if dims == None:
            dims = choose_mutate_index(filters)
            
        selected_filter = filters[dims[0]][dims[1]][dims[2]]
        
        # create new random filter to replace the selected filter
        # selected_filter = torch.tensor(np.random.rand(3,3), device=helper.device)
        
        # modify the entire layer / filters by a small amount
        # TODO: play around with lr multiplier on noise
        # TODO: implement broader mutation with low learning rate
        selected_filter += (torch.rand(selected_filter.shape[0], selected_filter.shape[1])*2.0-1.0)*mr

        # normalize entire filter so that values are between -1 and 1
        # selected_filter = (selected_filter/np.linalg.norm(selected_filter))*2
        
        # normalize just the values that are outside of -1, 1 range
        selected_filter[(selected_filter > 1) | (selected_filter < -1)] /= torch.amax(torch.absolute(selected_filter))
        
        filters[dims[0]][dims[1]][dims[2]] = selected_filter
        # print(selected_filter)
        return filters
    else:
        for i in range(len(filters)):
            mut = (torch.rand(filters[i].shape[0], filters[i].shape[1], filters[i].shape[2], filters[i].shape[3])*2-1.0)*mr
            # print(filters[i].shape)
            # print(mut.shape)
            filters[i] += mut
            divisor = torch.amax(torch.absolute(filters[i]))
            # condition = filters[i][(filters[i] > 1) | (filters[i] < -1)]
            # filters[i].where(condition, filters[i], filters[i] / divisor)
            filters[i][(filters[i] > 1) | (filters[i] < -1)] /= divisor
        return filters


def plot_mean_and_bootstrapped_ci_multiple(input_data = None, title = 'overall', name = "change this", x_label = "x", y_label = "y", x_mult=1, y_mult=1, save_name="", compute_CI=True, maximum_possible=None, show=None, sample_interval=None, legend_loc=None, alpha=1, y=None):
    """ 
     
    parameters:  
    input_data: (numpy array of numpy arrays of shape (max_k, num_repitions)) solution met
    name: numpy array of string names for legend 
    x_label: (string) x axis label 
    y_label: (string) y axis label 
     
    returns: 
    None 
    """ 
 
    generations = len(input_data[0])
 
    fig, ax = plt.subplots() 
    ax.set_xlabel(x_label) 
    ax.set_ylabel(y_label) 
    ax.set_title(title) 
    for i in range(len(input_data)): 
        CIs = [] 
        mean_values = [] 
        for j in range(generations): 
            mean_values.append(np.mean(input_data[i][j])) 
            if compute_CI:
                CIs.append(bootstrap.ci(input_data[i][j], statfunction=np.mean)) 
        mean_values=np.array(mean_values) 
 
        high = [] 
        low = [] 
        if compute_CI:
            for j in range(len(CIs)): 
                low.append(CIs[j][0]) 
                high.append(CIs[j][1]) 
 
        low = np.array(low) 
        high = np.array(high) 

        if type(y) == type(None):
            y = range(0, generations)
        if (sample_interval != None):
            y = np.array(y)*sample_interval 
        ax.plot(y, mean_values, label=name[i], alpha=alpha)
        if compute_CI:
            ax.fill_between(y, high, low, alpha=.2) 
        if legend_loc is not None:
            ax.legend(bbox_to_anchor=legend_loc['bbox'], loc=legend_loc['loc'], ncol=1)
        else:
            ax.legend()
    
    if maximum_possible:
        ax.hlines(y=maximum_possible, xmin=0, xmax=generations, linewidth=2, color='r', linestyle='--', label='best poss. acc.')
        ax.legend()

    if save_name != "":
        plt.savefig('plots/' + save_name)
    if show != None:
        plt.show()
    
def log(input):
    if glob_rank == 0:
        wandb.log(input)

def update_config():
    if glob_rank == 0:
        wandb.config.update(config)

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

def get_weights_from_ckpt(ckpt_path, network='conv6'):
    if network == 'vgg16':
        net = BigNet(num_classes=0, classnames=[], diversity = {'type': 'relative', 'ldop':'w_mean', 'pdop':'mean', 'k': -1, 'k_strat': 'closest'})
    else:
        net = Net(num_classes=0, classnames=[], diversity={'type': 'relative', 'ldop':'w_mean', 'pdop':'mean', 'k': -1, 'k_strat': 'closest'})
    net.load_from_checkpoint(ckpt_path)
    return net.get_filters()

def find_best_lr(trainer, net, data_module):
    trainer.auto_lr_find=True
    trainer.datamodule = data_module
    lr_finder = trainer.tuner.lr_find(net, datamodule=data_module, num_training=1000, min_lr=1e-12)
    import matplotlib.pyplot as plt
    fig = lr_finder.plot(suggest=True)
    fig.tight_layout()
    fig.savefig('lr_finder.png', dpi=300, format='png')
    fig.show()
    print(lr_finder.suggestion())
    input()
    exit()

def run(seed=True, rank=0):
    torch.multiprocessing.freeze_support()
    if seed:
        pl.seed_everything(42, workers=True)
    
    torch.cuda.empty_cache()
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global PATH
    PATH = './cifar_net.pth'
    global config
    config = {}
    wandb.login(key='e50cb709dc2bf04072661be1d9b46ec60d59e556')
    wandb.finish()
    os.environ["WANDB_START_METHOD"] = "thread"
    # TODO: could put if statement here to determine if we should be logging. This is only necessary once ddp is actually working correctly.
    global glob_rank
    glob_rank = rank
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:300"
    if glob_rank == 0:
    # if glob_rank > -1:
        if torch.cuda.is_available():
            force_cudnn_initialization()
        wandb.init(project="perturbed-initializations") # group='DDP'

    # warnings.filterwarnings('ignore', category=pl_bolts.utils.  .UnderReviewWarning)


if __name__ == '__main__':
    run()