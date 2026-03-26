# Visualize activations and filters for evolved and random filters

import matplotlib.pyplot as plt
import argparse
import helper_hpc as helper
import wandb
from net import Net
import numpy as np
from scipy.stats import ranksums
from tqdm import tqdm


# arguments
parser=argparse.ArgumentParser(description="Process some input files")
parser.add_argument('--run_ids_0', nargs='+', type=str, help="enter id for first list of wandb experiments to link config")
parser.add_argument('--run_ids_1', nargs='+', type=str, help="enter id for second list of wandb experiment to link config")
parser.add_argument('--epoch_range', nargs=2, type=int, help='range of values to consider from array of val_acc')
# parser.add_argument('--diversity', help='run ranksums for diversity instead of accuracy', action='store_true', default=False)
args = parser.parse_args()

def run():
    
    helper.run(seed=False)

    epoch_range = [0,None]
    if args.epoch_range:
        epoch_range[0] = int(args.epoch_range[0]*6.25)
        epoch_range[1] = int(args.epoch_range[1]*6.25)
    # if args.val_acc_range:
    #     epoch_range = args.val_acc_range

    # log variables to config

    mean_gradients = {}
    std_gradients = {}


    # for each run, fetch run data
    for i in tqdm(range(len(args.run_ids_0))):
        run_id = args.run_ids_0[i]
        api = wandb.Api()
        # runs = api.runs("jdonovan/perturbed-initializations", filters={"id": {"$in": [args.run_ids_0]}})
        # print(len(runs))
        run = api.run("jdonovan/perturbed-initializations/" + run_id)
        # search = 'val_acc' if not args.diversity else 'val_novelty'
        hist = run.scan_history()
        history = [row for row in hist]
        counts = {}
        histogram_intervals = {}
        mean_gradients[i] = {}
        std_gradients[i] = {}

         
        # for each layer in the network
        print(run_id)
        for layer in sorted([row for row in history[0]]):
            print(layer)
            # filter out layers that are not convolutional
            if ('gradient' in layer and 'weight' in layer and 'features' in layer) or ('gradient' in layer and 'conv' in layer and 'weight' in layer):
                # make human readable layer name
                layer_name = layer.split('weight')[0]
                layer_name = layer_name.replace('model.', '')
                layer_name = layer_name.replace('gradients/', '')
                print(layer_name)
                # if dictionaries do not alrady have this layer, create the index for it
                if counts != None and layer_name not in counts and counts.get(layer_name) == None:
                    counts[layer_name] = []
                    histogram_intervals[layer_name] = []

                # store some data for this layer across all gradient steps into variables
                grad_min = np.array([row[layer]['packedBins']['min'] for row in history if row[layer] != None])
                grad_size = np.array([row[layer]['packedBins']['size'] for row in history if row[layer] != None])
                grad_count = np.array([row[layer]['packedBins']['count'] for row in history if row[layer] != None])
                
                # get histogram intervals at each gradient step and add them to the dictionary for this layer
                for step in range(len(grad_min)):
                    histogram_intervals[layer_name].append(np.array([grad_min[step] + (grad_size[step] * j) for j in range(grad_count[step])]))

                # get counts of gradients that fall within these histogram values at each step    
                counts[layer_name] = np.array(np.array([row[layer]['values'] for row in history if row[layer] != None]))
                histogram_intervals[layer_name] = np.array(histogram_intervals[layer_name])
        for layer in counts.keys():
            mean_gradients[i][layer] = []
        for layer in counts.keys():
            mean_gradients[i][layer] = np.mean(counts[layer].astype(int)*histogram_intervals[layer], axis=1)

        for layer in counts.keys():
            std_gradients[i][layer] = []
        for layer in counts.keys():
            std_gradients[i][layer] = np.std(counts[layer].astype(int)*histogram_intervals[layer], axis=1)
        

        # for layer in list(counts.keys()):
        #     # if gradient == "gradients/model.features.41.weight":
        #     #     print(gradient, len([(i, row[gradient]) for i, row in enumerate(history) if row[gradient] != None]))
        #     # print([row[layer] for row in history if row[layer] != None][0])
        #     print(layer)
        #     grad_min = np.array([row[layer]['packedBins']['min'] for row in history if row[layer] != None])
        #     grad_size = np.array([row[layer]['packedBins']['size'] for row in history if row[layer] != None])
        #     grad_hist_len = [row[layer]['packedBins']['count'] for row in history if row[layer] != None][0]
        #     if counts[layer] == []:
        #         histogram_intervals[layer] = [grad_min[i] + (grad_size[i] * i) for i in range(len(grad_min))]
        #         counts[layer] = [row[layer]['values'] for row in history if row[layer] != None]
        #     else:
        #         histogram_intervals[layer] = histogram_intervals[layer] + np.array([grad_min[i] + (grad_size[i] * i) for i in range(len(grad_min))])
        #         counts[layer] = counts[layer] + np.array([row[layer]['values'] for row in history if row[layer] != None])
        #     ref_grad = layer

    mean_gradients_for_plotting = {}
    std_gradients_for_plotting = {}
    for layer in counts.keys():
        print(layer)
        mean_gradients_for_plotting[layer] = np.mean(np.array([mean_gradients[k][layer][epoch_range[0]:epoch_range[1]] for k in range(len(args.run_ids_0))]), axis=0)[epoch_range[0]:epoch_range[1]]
        print(np.mean(mean_gradients_for_plotting[layer]))
        if np.abs(np.mean(mean_gradients_for_plotting[layer])) > .1:
            plt.plot([x/6.25 for x in range(len(mean_gradients_for_plotting[layer]))], mean_gradients_for_plotting[layer], label=layer)
    plt.xlabel("Training Epoch")
    plt.ylabel('Mean Gradient Value')
    plt.title("Mean Gradient vs Training Epoch for {} (cifar100)".format(run.config['experiment_name']))
    plt.legend()
    plt.show()
        
    for layer in counts.keys():
        std_gradients_for_plotting[layer] = np.mean(np.array([std_gradients[k][layer][epoch_range[0]: epoch_range[1]] for k in range(len(args.run_ids_0))]), axis=0)[epoch_range[0]:epoch_range[1]]
        if np.abs(np.mean(mean_gradients_for_plotting[layer])) > .1:
            plt.plot([x/6.25 for x in range(len(std_gradients_for_plotting[layer]))], std_gradients_for_plotting[layer], label=layer)
    plt.xlabel("Training Epoch")
    plt.ylabel('Stand Deviation of Gradient Value')
    plt.title("Std of Gradient vs Training Epoch for {} (cifar100)".format(run.config['experiment_name']))
    plt.legend()
    plt.show()

    # mean_gradients = {}
    # for layer in counts.keys():
    #     for j in range(len(counts[layer])):
    #         print(counts[layer][j])
    #         print(histogram_intervals[layer][j])
    #         exit()
    #         mean_gradients[layer] = []
    # for layer in counts.keys():
    #     for j in range(len(counts[layer])):
    #         mean_gradients[layer].append(np.mean(np.array(counts[layer][j]).astype(int)*np.array(histogram_intervals[layer][j])/len(args.run_ids_0)))

    # std_gradients = {}
    # for layer in counts.keys():
    #     for j in range(len(counts[layer])):
    #         std_gradients[layer] = []
    # for layer in counts.keys():
    #     for j in range(len(counts[layer])):
    #         std_gradients[layer].append(np.mean(np.array(np.std(counts[layer][j]).astype(int)*np.array(histogram_intervals[layer][j]))))

    # mean_gradients = {key: [np.mean(gradients[key][index]) for index in range(len(gradient[key]))] for key in gradients.keys()}
    # std_gradients = {key: np.std(gradients[key]) for key in gradients.keys()}
        
    # print(counts)
    # print(mean_gradients)

    # for layer in counts.keys():
    #     plt.plot(mean_gradients[layer], label=layer)
    #     plt.legend()
    #     plt.show()

    # for layer in counts.keys():
    #     plt.plot(std_gradients[layer], label=layer)
    #     plt.legend()
    #     plt.show()
    # print(std_gradients)
    # print(len(gradients[ref_grad]))
        # print(history['gradients/model.features.41.weight'][5])
        # values = [row[search] for row in history if not np.isnan(row[search])]
        # print(len(values))
        # values_0.extend(values[epoch_range[0]:epoch_range[1]])

    # for i in range(len(args.run_ids_1)):
    #     run_id = args.run_ids_1[i]
    #     api = wandb.Api()
    #     run = api.run("jdonovan/perturbed-initializations/" + run_id)
    #     search = 'val_acc' if not args.diversity else 'val_novelty'
    #     history = run.scan_history(keys=[search])
    #     values = [row[search] for row in history if not np.isnan(row[search])]
    #     print(len(values))
    #     values_1.extend(values[epoch_range[0]: epoch_range[1]])



    # # run ranksums test
    # print(values_1, '\n', values_0)
    # print(ranksums(values_1, values_0))
    


if __name__ == '__main__':
    run()