import matplotlib.pyplot as plt
import argparse
import helper_hpc as helper
import wandb
from net import Net
import numpy as np
from scipy.stats import ranksums
from tqdm import tqdm

# Initialize API
api = wandb.Api()

# Define project
project_path = "jdonovan/perturbed-initializations"
project_path_old = "jdonovan/novel-feature-detectors"


# arguments
parser=argparse.ArgumentParser(description="Process some input files")
parser.add_argument('--config_filters', nargs='+', type=str, help="enter config dictionary name and values in format name:[values]")
parser.add_argument('--config_filters2', nargs='+', type=str, help="enter config filter criteria in format name:[values]")
parser.add_argument('--val_acc_range', nargs=2, type=int, help='range of values to consider from array of val_acc')
parser.add_argument('--diversity', help='run ranksums for diversity instead of accuracy', action='store_true', default=False)
args = parser.parse_args()

# --- Common Filtering Scenarios ---

def run():
    
    helper.run(seed=False)
    epoch_range = [0,None]
    if args.val_acc_range:
        epoch_range = args.val_acc_range
        
    def is_float_string(s):
        try:
            float(s)
            return True
        except (ValueError, TypeError):
            return False
    
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
    print(filters)
    # exit()
    runs = api.runs(
        project_path,        
        filters=filters
    )
    
    print(len(runs))
    
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
    print(filters2)
    
    runs2 = api.runs(
        project_path,    
        filters=filters2
    )
    
    print(len(runs2))
    
    
    # begin getting values
    values_0 = []
    values_1 = []
    
    print("getting first criteria experiments")
    for r in tqdm(runs):
        run_id = r.id
        api = wandb.Api()
        run = api.run(project_path + "/" + run_id)
        search = 'val_acc' if not args.diversity else 'val_novelty'
        history = run.scan_history(keys=[search])
        values = [row[search] for row in history if not np.isnan(row[search])]
        # print(len(values))
        values_0.extend(values[epoch_range[0]:epoch_range[1]])

    print("getting 2nd criteria experiments")
    for r in tqdm(runs2):
        run_id = r.id
        api = wandb.Api()
        run = api.run(project_path + "/" + run_id)
        search = 'val_acc' if not args.diversity else 'val_novelty'
        history = run.scan_history(keys=[search])
        values = [row[search] for row in history if not np.isnan(row[search])]
        # print(len(values))
        values_1.extend(values[epoch_range[0]: epoch_range[1]])
        
    print('num exps in 1:', len(values_0), 'num exps in 2:', len(values_1))
    
    print(values_0, '\n', values_1)
    print(ranksums(values_0, values_1))

    

if __name__ == '__main__':
    run()