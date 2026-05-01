import pickle
import argparse
import numpy as np
import wandb
import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import helper_hpc as helper


parser=argparse.ArgumentParser(description="Process some input files")
parser.add_argument('--input', help='pickle file input with metrics for plotting', default=None)
parser.add_argument('--ylabel', help='label for the y axis', default='Fitness')
parser.add_argument('--xlabel', help='label for the x axis', default='Generation')

parser.add_argument('--config_filters', nargs='+', type=str, help="enter config dictionary name and values in format name:[values]")
parser.add_argument('--config_filters2', nargs='+', type=str, help="enter config filter criteria in format name:[values]")
parser.add_argument('--metric', help='metric to compare in t-test', type=str, default='test_acc')
parser.add_argument('--metric2', help='metric to compare in t-test', type=str, default=None)

parser.add_argument('--x_label', help='label for x axis', default=None)
parser.add_argument('--y_label', help='label for y axis', default=None)
parser.add_argument('--title', help='Title for plot', type=str, default=None)

args = parser.parse_args()

def run():
    
    helper.run(seed=False)
    epoch_range = [0,None]
    # if args.val_acc_range:
    #     epoch_range = args.val_acc_range

    title=args.title if args.title is not None else f'{args.metric.capitalize()} by {args.metric2.capitalize()}'
    x_label=args.x_label if args.x_label is not None else args.metric2
    y_label=args.y_label if args.y_label is not None else args.metric
    
    def is_float_string(s):
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            return False
    
    project_path = "jdonovan/perturbed-initializations"
    api = wandb.Api()
    
    # parse filters and get first set of runs
    filters={"config.experiment_type": "training", "config.fixed_conv": False, "config.bn": True, "config.dataset": "cifar-100"}
    for filter_item in args.config_filters:
        filter_criteria_list = filter_item.split(":")[1].replace('[','').replace(']','').split(",")
        if all(item.isdigit() for item in filter_criteria_list):
            filter_criteria_list = [int(i) for i in filter_criteria_list]
        elif all(is_float_string(item) for item in filter_criteria_list):
            filter_criteria_list = [float(i) for i in filter_criteria_list]
        elif all(item.strip().lower() in ["true", "false"] for item in filter_criteria_list):
            filter_criteria_list = [i.lower() == "true" for i in filter_criteria_list]
        filters["config."+str(filter_item).split(":")[0]] = {"$in": filter_criteria_list}
    filters["summary_metrics."+str(args.metric)] = {"$nin": [None, "null"]}
    # print(filters)
    # exit()
    runs = api.runs(
        project_path,        
        filters=filters
    )
    
    # print(len(runs))
    
    # parse filters and get second set of runs
    filters2={"config.experiment_type": "training", "config.fixed_conv": False, "config.bn": True, "config.dataset": "cifar-100"}
    
    for filter_item in args.config_filters2:
        filter_criteria_list = filter_item.split(":")[1].replace('[','').replace(']','').split(",")
        if all(item.isdigit() for item in filter_criteria_list):
            filter_criteria_list = [int(i) for i in filter_criteria_list]
        elif all(is_float_string(item) for item in filter_criteria_list):
            filter_criteria_list = [float(i) for i in filter_criteria_list]
        elif all(item.strip().lower() in ["true", "false"] for item in filter_criteria_list):
            filter_criteria_list = [i.lower() == "true" for i in filter_criteria_list]
        filters2["config."+str(filter_item).split(":")[0]] = {"$in": filter_criteria_list}
    filters2["summary_metrics."+str(args.metric)] = {"$nin": [None, "null"]}
    # print(filters2)
    
    runs2 = api.runs(
        project_path,    
        filters=filters2
    )
    
    # print(len(runs2))
        
    print("getting experiment results")
    print('num exps in 1:', len(runs), 'num exps in 2:', len(runs2))
    x_values_0 = [run.summary[args.metric] for run in runs]
    x_values_1 = [run.summary[args.metric] for run in runs2]
    name_0 = [run.config['experiment_name'] for run in runs][0]
    name_1 = [run.config['experiment_name'] for run in runs2][0]
    if args.metric2 is not None:
        y_values_0 = [run.summary[args.metric2] for run in runs]
        y_values_1 = [run.summary[args.metric2] for run in runs2]
        print(list(zip(x_values_0, y_values_0)), '\n', list(zip(x_values_1, y_values_1)))   
    
    
        helper.plot_mean_and_bootstrapped_ci_multiple(input_data=np.array([np.array(x_values_0), np.array(x_values_1)]), y=np.array([np.array(y_values_0), np.array(y_values_1)]), title=title, name=[name_0, name_1], x_label=x_label, y_label=y_label, compute_CI=False, show=True, plot_type='scatter')
    else:
        helper.create_violin_plot([x_values_0, x_values_1], [name_0, name_1], title=title, ylabel=y_label)


if __name__ == "__main__":
    if args.input is not None:
        with open(args.input, 'rb') as f:
            pickled_metrics = pickle.load(f)
            print(pickled_metrics)

        if type(pickled_metrics) == dict:
            helper.plot_mean_and_bootstrapped_ci_multiple(input_data=[np.transpose(x) for k, x in pickled_metrics.items()], name=[k for k,x in pickled_metrics.items()], x_label="Generation", y_label="Fitness", compute_CI=True, show=True)

    else:
        run()