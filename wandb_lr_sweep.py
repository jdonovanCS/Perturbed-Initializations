import wandb
import helper_hpc as helper
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import torch
import argparse
from functools import partial


#args
parser=argparse.ArgumentParser(description="Process some input files")
parser.add_argument('--experiment_name', help='experiment name for saving data related to training')
parser.add_argument('--rand_tech', help='which random technique is used to initialize network weights', type=str, default=None)
parser.add_argument('--dataset', help='which dataset should be used for training metric, choices are: cifar-10, cifar-100, tinyimagenet', default='cifar-100')
parser.add_argument('--batch_size', help="batch size for training", type=int, default=64)
parser.add_argument('--num_workers', help='number of workers for training', default=np.inf, type=int)
parser.add_argument('--use_scheduler', help='if using cosine annealling optimizer scheduling is preferred', action='store_true')
parser.add_argument('--early_stopping', help='if early stopping based on patience of val loss is desired use this', action='store_true')
parser.add_argument('--epochs', help="Number of epochos to train for", type=int, default=256)
parser.add_argument('--no_bn', help='Train networks without batchnorm layers', action='store_true')

parser.add_argument('--method', help='Sweep method (ie. random, bayes, grid)',type=str, default='bayes')
parser.add_argument('--lr_min', help='Sweep method (ie. random, bayes, grid)',type=float, default=.000001)
parser.add_argument('--lr_max', help='Sweep method (ie. random, bayes, grid)',type=float, default=.01)
parser.add_argument('--count', help='Sweep method (ie. random, bayes, grid)', type=int, default=20)




args = parser.parse_args()

def run():
    torch.multiprocessing.freeze_support()
    
    stored_filters = {}
    
    experiment_name = args.experiment_name
    
    filename = 'output/' + experiment_name + f'/solutions_over_time_{args.rand_tech}.npy'
    
    # get filters from numpy file
    np_load_old = partial(np.load)
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    stored_filters = np.load(filename)
    np.load = np_load_old
    
    helper.run(seed=False, rank=0)
    
    data_module = helper.get_data_module(args.dataset, args.batch_size, args.num_workers)
    data_module.prepare_data()
    data_module.setup()

    epochs = args.epochs
    
    lr = wandb.config.learning_rate
    print(lr)
    helper.config['bn'] = not args.no_bn
    helper.config['dataset'] = args.dataset.lower()
    helper.config['batch_size'] = args.batch_size
    helper.config['lr'] = lr
    helper.config['experiment_name'] = args.experiment_name
    helper.config['experiment_type'] = 'lr_sweep'
    helper.config['early_stopping'] = args.early_stopping
    helper.config['scheduler'] = args.use_scheduler
    helper.update_config()
    
    scaled = False
    if len(stored_filters[0][0]) > 6:
        scaled = True
    
    helper.train_network(data_module=data_module, filters=stored_filters[0][0], epochs=epochs, lr=lr, save_path=None, fixed_conv=False, novelty_interval=0, val_interval=1, diversity=None, scaled=scaled, devices=1, save_interval=None, bn=not args.no_bn, log_activations=False, early_stopping=args.early_stopping, use_scheduler=args.use_scheduler)
    
    
    
if __name__ == "__main__":
    sweep_configuration = {
        'method': args.method,
        'metric': {'name': 'val_acc', 'goal': 'maximize'},
        'parameters': {
            'learning_rate': {
                'min': args.lr_min,
                'max': args.lr_max
            }
        }
    }
    
    sweep_id = wandb.sweep(sweep_configuration, project='perturbed-initializations')
    
    wandb.agent(sweep_id, function=run, count=args.count)
    
    