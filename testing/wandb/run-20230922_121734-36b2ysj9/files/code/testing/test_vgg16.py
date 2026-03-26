from pytorch_lightning.loggers import WandbLogger
import os
import wandb
import warnings
warnings.simplefilter('ignore')

# for net
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision.models as models
from net_copy import Net as exists
import numpy as np
# import helper_hpc as helper
from functools import partial

# for data module
from typing import Any, Callable, Optional, Sequence, Union
from torchvision.datasets import CIFAR100
from pl_bolts.datamodules import CIFAR10DataModule
from torchvision import transforms

class Net(pl.LightningModule):
    def __init__(self, num_classes=10, classnames=None, diversity=None, lr=.1):
        super().__init__()

        self.save_hyperparameters()
        self.num_classes = num_classes

        self.classnames = classnames
        self.diversity = diversity
        # self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        # self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.lr = lr

        self.model = models.vgg16_bn(pretrained=False)# num_classes=self.num_classes)
        self.model.classifier[6] = nn.Linear(in_features=4096, out_features=self.num_classes)
        # self.model = exists(self.num_classes, self.classnames, self.diversity, self.lr)

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        logits = self.forward(x)
        # # print(torch.argmax(logits,1), y)

        loss = self.cross_entropy_loss(logits, y)

        y_hat = torch.argmax(logits, 1)
        acc = torch.sum(y==y_hat)/(len(y)*1.0)

        # # self.train_acc(logits, y)

        # # log loss and acc
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        batch_dictionary={
	            "train_loss": loss, "train_acc": acc, 'loss': loss
	        }
        
        return batch_dictionary
        return self.model.training_step(train_batch, batch_idx)

    def training_epoch_end(self,outputs):
        avg_loss = torch.stack([x['train_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['train_acc'] for x in outputs]).mean()
        
        self.log('train_loss_epoch', avg_loss)
        self.log('train_acc_epoch', avg_acc)
        # self.model.training_epoch_end(outputs)
    
    def validation_step(self, val_batch, batch_idx):
        with torch.no_grad():
            x, y = val_batch
            
            logits = self.forward(x)

            loss = self.cross_entropy_loss(logits, y)

            # self.valid_acc(logits, y)
            y_hat = torch.argmax(logits, 1)
            acc = torch.sum(y==y_hat)/(len(y)*1.0)

            self.log('val_loss', loss)
            self.log('val_acc', acc)
            batch_dictionary = {'val_loss': loss, 
                                'val_acc': acc
                                }
        return batch_dictionary
        # return self.model.validation_step(val_batch, batch_idx)
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        self.log('val_loss_epoch', avg_loss)
        self.log('val_acc_epoch', avg_acc)
        # self.log('val_acc_epoch', self.valid_acc, sync_dist=True)
        # self.model.validation_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        # optimizer = self.model.configure_optimizers()
        return optimizer
    
    def cross_entropy_loss(self, logits, labels):
        # return F.nll_loss(logits, labels)
        return F.cross_entropy(logits, labels)
        # self.model.cross_entropy_loss(logits, labels)

    # def train_dataloader(self):
    #     d = CIFAR100DataModule(batch_size=64, data_dir="data/", num_workers=min(2, os.cpu_count()), pin_memory=True)
    #     d.prepare_data()
    #     d.setup()
    #     return d.train_dataloader()


class CIFAR100DataModule(CIFAR10DataModule):

    name = "cifar100"
    dataset_cls = CIFAR100
    dims = (3, 32, 32)

    def __init__(self,
        data_dir: Optional[str] = None,
        val_split: Union[int, float] = 0.2,
        num_workers: int = 0,
        normalize: bool = False,
        batch_size: int = 32,
        seed: int = 42,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:

        """
        Args:
            data_dir: Where to save/load the data
            val_split: Percent (float) or number (int) of samples to use for the validation split
            num_workers: How many workers to use for loading data
            normalize: If true applies image normalize
            batch_size: How many samples per batch to load
            seed: Random seed to be used for train/val/test splits
            shuffle: If true shuffles the train data every epoch
            pin_memory: If true, the data loader will copy Tensors into CUDA pinned memory before
                        returning them
            drop_last: If true drops the last incomplete batch
        """

        super().__init__(  # type: ignore[misc]
            data_dir=data_dir,
            val_split=val_split,
            num_workers=num_workers,
            normalize=normalize,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            pin_memory=pin_memory,
            drop_last=drop_last,
            *args,
            **kwargs,
        )

    @property
    def num_classes(self) -> int:
        """
        Return:
            100
        """
        return 100

    def default_transforms(self) -> Callable:
        stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
        transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32,padding=4,padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
        ])
        return transform

def train_vgg16(data_module, epochs=120, lr=5e-5, val_interval=4):

    classnames = list(data_module.dataset_test.classes)
    diversity = {'type': 'relative', 'pdop': 'mean', 'ldop': 'w_mean', 'k':-1, 'k_strat': 'closest'}
    net = Net(num_classes=data_module.num_classes, classnames=classnames, diversity=diversity, lr=lr)
    np_load_old = partial(np.load)
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    stored_filters = np.load('output/relative diversity scaled/solutions_over_time_fitness.npy')
    stored_filters = stored_filters[0][49]
    count = 0
    for m in net.model.modules():
        if isinstance(m, (torch.nn.Conv2d)):
            # print(m)
            # print(stored_filters[count].shape)
            m.weight.data = stored_filters[count]
            count += 1

    np.load = np_load_old
    # helper.normalize(net.model)
    
    wandb_logger = WandbLogger(log_model=True)
    trainer = pl.Trainer(max_epochs=epochs, logger=wandb_logger, check_val_every_n_epoch=val_interval, accelerator="gpu")#, devices=1, plugins=DDPPlugin(find_unused_parameters=False))
    
    # lr_finder = trainer.tuner.lr_find(net)
    # fig = lr_finder.plot(suggest=True)
    # print(lr_finder.suggestion())
    # return lr_finder.suggestion()

    wandb_logger.watch(net, log="all")
    trainer.fit(net, datamodule=data_module)

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

    data_module = CIFAR100DataModule(batch_size=64, data_dir="../data/", num_workers=min(2, os.cpu_count()), pin_memory=True)
    data_module.prepare_data()
    data_module.setup()

    # suggestions = []
    for i in range(10):
        wandb.finish()
        wandb.init(project="perturbed-initializations")
        train_vgg16(data_module)
        # suggestions.append(train_vgg16(data_module))
    # print(suggestions)

    # lr_suggestions
    # normal: [0.01445439770745928, 0.05248074602497723, 0.005754399373371567, 3.9810717055349735e-05, 0.01445439770745928, 1.9054607179632464e-05, 0.006918309709189364, 5.7543993733715664e-05, 0.030199517204020192, 1.584893192461114e-05]
    # uniform: [0.003981071705534969, 0.003981071705534969, 0.00478630092322638, 0.030199517204020192, 0.006918309709189364, 1.9054607179632464e-05, 0.012022644346174132, 0.003981071705534969, 1.9054607179632464e-05, 0.02089296130854041]