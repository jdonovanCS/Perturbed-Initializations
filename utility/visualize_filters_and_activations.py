# Visualize activations and filters for evolved and random filters

import matplotlib.pyplot as plt
import argparse
import helper_hpc as helper
import wandb
import pickle
from net import Net
import numpy as np
import os
import time
import torch


# arguments
parser=argparse.ArgumentParser(description="Process some input files")
parser.add_argument('--run_id', help="enter id for wandb experiment to link config")
parser.add_argument('--training_interval', help='How often should the network be trained. Values should be supplied as a fraction and will relate to the generations from evolution' +
'For example if 1 is given the filters generated from the final generation of evolution will be the only ones trained. If 0.5 is given then the halfway point of evolutionary generations and the final generation will be trained. ' +
'If 0 is given, the filters from every generation will be trained', type=float, default=None)
args = parser.parse_args()

def run():
    
    helper.run(seed=False)

    # log variables to config
    run_id = args.run_id
    api = wandb.Api()
    run = api.run("jdonovan/perturbed-initializations/" + run_id)
    helper.config['experiment_type'] = 'visualization'
    filters = {}
    print(run.config)
    for k, v in run.config.items():
        helper.config[k] = v
        filters["config." + k] = v
    helper.update_config()


    # narrow down exactly which run we want the filters from
    runs = api.runs(path="jdonovan/perturbed-initializations", filters=filters, order="created_at", per_page=10)
    count = 0
    spec_run_num = 0
    for r in runs:
        if r.path == run.path:
            spec_run_num=count
        count+=1
    
    # get filters for this run

    if run.config['evo']:
        filename = "output/" + run.config['experiment_name'] + "/solutions_over_time.pickle"
        training_interval = 1.
    else:
        filename = 'output/' + run.config['experiment_name'] + '/random_gen_solutions.pickle'
        training_interval = .2
        

    if args.training_interval != None:
        training_interval = args.training_interval
    
    with open(filename, 'rb') as f:
        pickled_filters = pickle.load(f)

    for k in pickled_filters:
        for run_num in range(len(pickled_filters[k])):
            for i in range(len(pickled_filters[k][run_num])):
                if training_interval == 0 or i*1.0 == [(len(pickled_filters[k][run_num])/(1/training_interval)*j)-1 for j in range(1, int(1/training_interval)+1)][spec_run_num]:
                    filters = pickled_filters[k][run_num][i]
                    if run.config['experiment_type']=='training':
                        filename = "trained_models/trained/conv{}_e{}_n{}_r{}_g{}.pth".format(not run.config['fixed_conv'], run.config['experiment_name'], k, run_num, i)
                        while os.path.isdir(filename) and len(os.listdir(filename)) > 0:
                            filename += "/" + os.listdir(filename)[0]
                        net = Net.load_from_checkpoint(filename)
                        filters = net.get_filters()

    # get data to push into network
    data_module = helper.get_data_module(run.config['dataset'], 1)
    data_module.prepare_data()
    data_module.setup()
    
    # visualize filters
    for layer in filters:
        print(layer.shape)
        plt.figure()
        for i in range(len(layer)):
            for j in range(len(layer[i])):
                values = np.array(255*((layer[i][j] + 1) /2)).astype(np.int64)
                rows = len(layer[i])
                cols = len(layer)
                plt.subplot(rows, cols, i*len(layer[0]) + (j+1))
                plt.imshow(values, cmap="gray", vmin = 0, vmax = 255,interpolation='none')
            # values = np.array(255*((layer[i] + 1) /2)).astype(np.int64)
            # if len(np.where(values > 255)) > 0:
            #     print(values)
            #     print(layer[i])
            # rows = cols = int(np.ceil(np.sqrt(len(layer))))
            # plt.subplot(rows, cols, i+1)
            # plt.imshow(values.transpose(1, 2, 0), vmin = 0, vmax = 255,interpolation='none')
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        plt.show()

    # visualize activations
    if os.path.isfile('tensor.pt'):
        x = torch.load('tensor.pt')
    else:
        data_module = helper.get_data_module(run.config['evo_dataset_for_novelty'], batch_size=1)
        data_module.prepare_data()
        data_module.setup()
        classnames = list(data_module.dataset_test.classes)
        net = Net(num_classes=len(classnames), classnames=classnames)
        net.set_filters = filters
        batch = next(iter(data_module.val_dataloader()))
        x, y = batch
        torch.save(x, 'tensor.pt')
        x = torch.load('tensor.pt')
    net.forward(x, get_activations=True)
    activations = net.activations
    l = []
    for layer in activations: #layers
        print(len(activations[layer]))
        activations[layer][0] = activations[layer][0].detach().cpu().numpy()
        print(len(activations[layer][0]))
        print(len(activations[layer][0][0]))
        print(len(activations[layer][0][0][0]))
        for image in range (len(activations[layer][0])): # images in batch
            for channel in range(len(activations[layer][0][image])): # channels in activation
                values = np.array(activations[layer][0][image][channel]).astype(np.int64)
                if len(np.where(values > 255)) > 0:
                    print(values)
                    print(activations[layer][0][image])
                rows = cols = int(np.ceil(np.sqrt(len(activations[layer][0][image]))))
                plt.subplot(rows, cols, channel+1)
                plt.imshow(values, interpolation='none')
            plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
            plt.show()
    exit()
    


if __name__ == '__main__':
    run()