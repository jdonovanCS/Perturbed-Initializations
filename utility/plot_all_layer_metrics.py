import argparse
import numpy as np
import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import helper_hpc as helper
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from tqdm import tqdm

parser=argparse.ArgumentParser(description="Process some input files")

parser.add_argument('--config_filters', nargs='+', type=str, help="enter config dictionary name and values in format name:[values]")
parser.add_argument('--config_filters2', nargs='+', type=str, help="enter config filter criteria in format name:[values]")
parser.add_argument('--metric', help='metric to compare in t-test', type=str, default='test_acc')
parser.add_argument('--metric2', help='metric to compare in t-test', type=str, default=None)

parser.add_argument('--epoch', help='epoch to compare metrics at', type=int, default=None)

parser.add_argument('--x_label', help='label for x axis', default=None)
parser.add_argument('--y_label', help='label for y axis', default=None)
parser.add_argument('--title', help='Title for plot', type=str, default=None)

args = parser.parse_args()

def run():
    helper.run(seed=False)
    
    
    ## Get all data for specified runs

    # Create filters for specified experiments

    # Get runs for specified filters
    # parse filters and get first set of runs
    filters=helper.create_wandb_filters(args.config_filters)

    runs = helper.get_wandb_runs(filters)
    
    # parse filters and get second set of runs
    filters2 = helper.create_wandb_filters(args.config_filters2, args.metric)
    
    runs2 = helper.get_wandb_runs(filters2)
        
    print("getting experiment results")
    print('num exps in 1:', len(runs), 'num exps in 2:', len(runs2))

    # For each layer in network for runs, get metric values
    data_dict = {'test_acc': [], 'epoch': [], 'average_activation_map_correlation': [], 'average_activation_map_cosine_distance': [], 'average_activation_map_covariance': [], 'average_activation': []}
    layer_num = 0
    for run in runs:
        for key in run.summary.keys():
            # get number of layers by looking at activation_layernum_epoch keys
            if 'activation_map_correlation' in key and 'epoch' in key:
                # get max layer number
                layer_num = max(layer_num, int("".join([char for char in (key.split('_')[2]) if char.isdigit()])))
                data_dict['activation_map_correlation'+str(layer_num)] = []
                data_dict['activation_'+str(layer_num)] = []
                data_dict['activation_map_cosine_distance'+str(layer_num)] = []
                data_dict['activation_map_covariance'+str(layer_num)] = []
    for run in tqdm(runs):
        if args.epoch is not None:
            key_list = ['activation_map_correlation'+str(i)+'_epoch' for i in range(1, layer_num+1)] + ['activation_'+str(i)+'_epoch' for i in range(1, layer_num+1)] + ['activation_map_cosine_distance'+str(i)+'_epoch' for i in range(1, layer_num+1)] + ['activation_map_covariance'+str(i)+'_epoch' for i in range(1, layer_num+1)]
            history = run.scan_history(keys=key_list)
            history = [row for row in history if all(not np.isnan(row[key]) for key in key_list)]
        for i in range(1, layer_num+1):
            if args.epoch is None:
                data_dict['activation_map_correlation'+str(i)].append(run.summary['activation_map_correlation'+str(i)+'_epoch'])
                data_dict['activation_'+str(i)].append(run.summary['activation_'+str(i)+'_epoch'])
                data_dict['activation_map_cosine_distance'+str(i)].append(run.summary['activation_map_cosine_distance'+str(i)+'_epoch'])
                data_dict['activation_map_covariance'+str(i)].append(run.summary['activation_map_covariance'+str(i)+'_epoch'])
            else:
                data_dict['activation_map_correlation'+str(i)].append([row['activation_map_correlation'+str(i)+'_epoch'] for row in history][args.epoch])
                data_dict['activation_'+str(i)].append([row['activation_'+str(i)+'_epoch'] for row in history][args.epoch])
                data_dict['activation_map_cosine_distance'+str(i)].append([row['activation_map_cosine_distance'+str(i)+'_epoch'] for row in history][args.epoch])
                data_dict['activation_map_covariance'+str(i)].append([row['activation_map_covariance'+str(i)+'_epoch'] for row in history][args.epoch])

        data_dict['test_acc'].append(run.summary['test_acc'])
        data_dict['epoch'].append(run.summary['epoch'])
        data_dict['average_activation_map_correlation'].append(np.mean([data_dict['activation_map_correlation'+str(i)] for i in range(1, layer_num+1)]))
        data_dict['average_activation_map_cosine_distance'].append(np.mean([data_dict['activation_map_cosine_distance'+str(i)] for i in range(1, layer_num+1)]))
        data_dict['average_activation_map_covariance'].append(np.mean([data_dict['activation_map_covariance'+str(i)] for i in range(1, layer_num+1)]))
        data_dict['average_activation'].append(np.mean([data_dict['activation_'+str(i)] for i in range(1, layer_num+1)]))
            

    # 1. Load or create your data
    df = pd.DataFrame(data_dict)

    # 2. Run Pearson correlation (Pearson is the default)
    correlation_matrix = df.corr(method='pearson')

    # print(correlation_matrix)
    print(correlation_matrix['test_acc'].abs().sort_values(ascending=False))
    # average activation has the highest correlation with test acc
    # maybe we should evovle for lower average activation? and diverse features?
    # but this is for features at the end of training
    print(scipy.stats.pearsonr(df['test_acc'], df['average_activation']))
    print(scipy.stats.pearsonr(df['test_acc'], df['average_activation_map_correlation']))
    print(scipy.stats.pearsonr(df['test_acc'], df['average_activation_map_cosine_distance']))
    print(scipy.stats.pearsonr(df['test_acc'], df['average_activation_map_covariance']))
    print(scipy.stats.pearsonr(df['test_acc'], df[f'activation_{layer_num}']))

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix[['test_acc']].abs().sort_values(by='test_acc'), annot=True, cmap='coolwarm', fmt=".2f")

    # Add titles and labels for clarity
    plt.title('Pearson Correlation Matrix')
    plt.show()

if __name__ == "__main__":
    run()

    