import helper_hpc as helper
import pytorch_lightning as pl
from net import Net
import torch

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    dm = helper.get_data_module('cifar-100', 64, workers=3)
    dm.prepare_data()
    dm.setup()
    net = Net(num_classes=dm.num_classes, classnames=dm.dataset_test.classes, diversity=None, lr=0.00025, bn=True, log_activations=True, use_scheduler=True)
    trainer=pl.Trainer()
    trainer.test(net, datamodule=dm)

