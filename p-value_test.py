# Visualize activations and filters for evolved and random filters

import matplotlib.pyplot as plt
import argparse
import helper_hpc as helper
import wandb
from net import Net
import numpy as np
from scipy.stats import ranksums


# arguments
parser=argparse.ArgumentParser(description="Process some input files")
parser.add_argument('--run_ids_0', nargs='+', type=str, help="enter id for first list of wandb experiments to link config")
parser.add_argument('--run_ids_1', nargs='+', type=str, help="enter id for second list of wandb experiment to link config")
parser.add_argument('--val_acc_range', nargs=2, type=int, help='range of values to consider from array of val_acc')
parser.add_argument('--diversity', help='run ranksums for diversity instead of accuracy', action='store_true', default=False)
args = parser.parse_args()

def run():
    
    helper.run(seed=False)

    epoch_range = [0,None]
    if args.val_acc_range:
        epoch_range = args.val_acc_range

    values_0 = []
    values_1 = []
    

    # log variables to config
    for i in range(len(args.run_ids_0)):
        run_id = args.run_ids_0[i]
        api = wandb.Api()
        run = api.run("jdonovan/perturbed-initializations/" + run_id)
        search = 'val_acc' if not args.diversity else 'val_novelty'
        history = run.scan_history(keys=[search])
        values = [row[search] for row in history if not np.isnan(row[search])]
        print(len(values))
        values_0.extend(values[epoch_range[0]:epoch_range[1]])

    for i in range(len(args.run_ids_1)):
        run_id = args.run_ids_1[i]
        api = wandb.Api()
        run = api.run("jdonovan/perturbed-initializations/" + run_id)
        search = 'val_acc' if not args.diversity else 'val_novelty'
        history = run.scan_history(keys=[search])
        values = [row[search] for row in history if not np.isnan(row[search])]
        print(len(values))
        values_1.extend(values[epoch_range[0]: epoch_range[1]])



    # run ranksums test
    print(values_1, '\n', values_0)
    print(ranksums(values_1, values_0))
    


if __name__ == '__main__':
    run()