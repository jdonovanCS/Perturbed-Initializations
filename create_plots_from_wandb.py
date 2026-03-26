import wandb
import numpy as np
import argparse
import matplotlib.pyplot as plt
import scikits.bootstrap as bootstrap
import helper_hpc as helper

parser=argparse.ArgumentParser(description="Process some inputs")

# experiment params
# parser.add_argument('--filters', help='filters to use to get runs', default=None, type=str)
# parser.add_argument('--keys', nargs='+', help="Specify which keys to pay attention to", default=None, type=str)
parser.add_argument('--title', help='title of plot', default='')
parser.add_argument('--x_label', help='label for x axis', default='Epochs')
parser.add_argument('--x_mult', nargs='*', help='multipler for x values in plot', type=int, default=[4])
parser.add_argument('--y_mult', help='multiplier for y values in plot', type=int, default=1)
parser.add_argument('--y_label', help='label for y axis', default='Val Acc')
parser.add_argument('--save_name', help='where to save the plot', default='')
parser.add_argument('--exp_names_chart', nargs='*', help='names of experiments as seen on plot', default=[])
parser.add_argument('--x_min', help='limit the x value to be between x_min and x_max', default=None, type=int)
parser.add_argument('--x_max', help='limit the x value to be between x_min and x_max', default=None, type=int)
parser.add_argument('--legend_loc', help='where to anchor legend positionally', type=str)
parser.add_argument('--legend_x', help='x anchor for legend', type=float)
parser.add_argument('--legend_y', help='y anchor for legend', type=float)
parser.add_argument('--legend_width', help='set legend width', type=float)
parser.add_argument('--legend_height', help='set legend height', type=float)
parser.add_argument('--seperator', help='how to seperate runs shown in plot', default=None)


args = parser.parse_args()
print(args)
if args.legend_x and args.legend_y and args.legend_loc:
    if args.legend_width and args.legend_height:
        legend_loc = {'bbox': (args.legend_x, args.legend_y, args.legend_width, args.legend_height), 'loc': args.legend_loc}
    else:
        legend_loc={'bbox': (args.legend_x, args.legend_y), 'loc': args.legend_loc}
else:
    legend_loc=None
experiment_names = ["relative diversity", "relative diversity mr.5", "relative diversity broadmut mr1", "relative diversity broadmut mr.5", "relative diversity intended", "relative diversity intended mr.5"]
generations = [49, 39, 29, 19, 9]
args.filters = {"config.experiment_name": {"$in": experiment_names}, 
                "config.experiment_type": "evolution", 
                # "config.fixed_conv": {"$nin": [True]}, 
                # "config.dataset": {"$in": ["cifar10"]},# "cifar-100"]}, 
                # "config.generation": {"$nin": [149, 4, 2, 3, 1, 24, 8, 6, 34, 5, 44, 7, 14, 129, 116, 51, 103, 77, 25, 90, 12, 38, 64, 39, 29, 19, 9]}, 
                # "config.lr": {"$nin": [.01, .0002, .005]},
                'created_at': {'$gt': '2023-06-20T20'}}
args.keys = ['best_individual_fitness']

if args.keys[0] == 'val_acc_epoch':
    args.legend_loc = None
    args.x_max = 100
if args.keys[0] == 'val_novelty':
    legend_loc = {'bbox': (1.5,1.5), 'loc': 'center right'}
    args.y_label = 'Diversity'
if args.keys[0] == 'best_individual_fitness':
    args.legend_loc = None
    args.y_label = 'Fitness'
    args.x_label = 'Gens'
    args.x_mult = [1]

# Get data from wandb
api = wandb.Api(timeout=30)
runs = api.runs("jdonovan/perturbed-initializations", filters=args.filters)
# {"$and": [{'created_at': {'$gt': '2023-06-20T20'}}]}

# process and store that data
history = {}
data = {}
config = {}

if args.seperator == 'generation':
    history.update({e: {} for e in generations})
    data.update({e: {} for e in generations})
    config.update({e: {} for e in generations})
    experiment_names.remove('relative diversity')

history.update({e: {} for e in experiment_names})
data.update({e: {} for e in experiment_names})
config.update({e: {} for e in experiment_names})


x_set = False
for run in runs:
    print(run.id)
    if run.config['experiment_name'] == 'relative diversity' and args.seperator != None:
        seperator = args.seperator
    else:
        seperator = 'experiment_name'
    history[run.config[seperator]][run.id] = run.scan_history(keys=args.keys)
    data[run.config[seperator]][run.id] = {}
    for i, k in enumerate(args.keys):
        data[run.config[seperator]][run.id][k] = [row[k] for row in history[run.config[seperator]][run.id]]
        if not x_set:
            args.x_min = 0 if args.x_min == None else args.x_min//args.x_mult[i]+1 if len(args.x_mult) > i else args.x_min
            args.x_max = len(data[run.config[seperator]][run.id][k]) if args.x_max == None else args.x_max//args.x_mult[i]+1 if len(args.x_mult) > i else args.x_max
            x_set=True
        data[run.config[seperator]][run.id][k] = data[run.config[seperator]][run.id][k][args.x_min:args.x_max]
    config[run.id] = run.config
print(data.keys())
print(data)
# plot the data
plt.rcParams.update({'font.size': 22, "figure.figsize": (7, 6)})
for i, k in enumerate(args.keys):
    input_data = {}
    for experiment_name in [d for d in data.keys()]:
        input_data[experiment_name] = [d2[k] for d2 in data[experiment_name].values()]
    print(np.array(list(input_data.values())).shape, input_data.keys(), args.exp_names_chart)
    helper.plot_mean_and_bootstrapped_ci_multiple(input_data=[np.transpose(x) for k, x in input_data.items()], name=args.exp_names_chart if len(args.exp_names_chart) == len(input_data.keys()) else list(input_data.keys()), title=args.title, x_label=args.x_label, y_label=args.y_label, save_name=args.save_name, show=True, sample_interval=(args.x_mult[i] if len(args.x_mult)>i else 1), legend_loc=legend_loc)