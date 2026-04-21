import os
import numpy as np
import warnings
warnings.filterwarnings('ignore') # Danger, Will Robinson! (not a scalable hack, and may surpress other helpful warning other than for ill-conditioned bootstrapped CI distributions)
import helper_hpc as helper
import torch
import argparse
from functools import partial

parser=argparse.ArgumentParser(description="Process some input files")
parser.add_argument('--dataset', help='which dataset should be used for training metric, choices are: cifar-10, cifar-100, tinyimagenet', default='cifar-100')
parser.add_argument('--fixed_conv', help='Should the convolutional layers stay fixed, or alternatively be trained', action='store_true')
parser.add_argument('--training_interval', help='How often should the network be trained. Values should be supplied as a fraction and will relate to the generations from evolution' +
'For example if 1 is given the filters generated from the final generation of evolution will be the only ones trained. If 0.5 is given then the halfway point of evolutionary generations and the final generation will be trained. ' +
'If 0 is given, the filters from every generation will be trained', type=float, default=1.)
parser.add_argument('--epochs', help="Number of epochos to train for", type=int, default=256)
parser.add_argument('--devices', help='number of gpus to use', default=1, type=int)
parser.add_argument('--local_rank', metavar="N", help='if using ddp and multiple gpus, we only want to collect metrics once, input 0 here if using ddp and multi gpus', default=0, type=int)

# parser.add_argument('--rand_norm', action='store_true')
parser.add_argument('--gram-schmidt', help='gram-schmidt used to orthonormalize filters', action='store_true')
parser.add_argument('--novelty_interval', help='How often should a novelty score be captured during training?', default=0)
parser.add_argument('--val_accuracy_interval', help='How often should test accuracy be assessed during training?', default=1)
parser.add_argument('--batch_size', help="batch size for training", type=int, default=64)
parser.add_argument('--lr', help='Learning rate for training', default=.001, type=float)
parser.add_argument('--save_interval', help='How often (in epochs) should the model checkpoint be saved', default=None, type=int)
parser.add_argument('--use_scheduler', help='if using cosine annealling optimizer scheduling is preferred', action='store_true')
parser.add_argument('--early_stopping', help='if early stopping based on patience of val loss is desired use this', action='store_true')
parser.add_argument('--no_bn', help='Train networks without batchnorm layers', action='store_true')
parser.add_argument('--log_activations', help="Log activation values as the network is trained", action='store_true')

# used to link to evolution
parser.add_argument('--experiment_name', help='experiment name for saving data related to training')
parser.add_argument('--rand_tech', help='which random technique is used to initialize network weights', type=str, default=None)
# don't need any of the below for comparisons since I can link with the above experiment name.

# used to link to autoencoder
parser.add_argument('--ae', help="if pretrained using ae include this tag", action='store_true', default=False)

# Options for flexibility
parser.add_argument('--unique_id', help='if a unique id is associated with the file the solution is stored in give it here.', default="", type=str)
parser.add_argument('--skip', default=0, help='skip the first n models to train, used mostly when a run fails partway through', type=int)
parser.add_argument('--inner_skip', default=0, help='skip the first n models to train withint a specific run of evolution (only applies if  training interval < 1.)', type=int)
parser.add_argument('--stop_after', default=np.inf, help='stop after the first n models', type=int)
parser.add_argument('--inner_stop_after', default=np.inf, help='stop after the first n models for a specific run of evolution (only applies if training interval < 1.)', type=int)
parser.add_argument('--num_workers', help='number of workers for training', default=np.inf, type=int)

# Options for measuring diversity over training time
parser.add_argument('--diversity_type', type=str, default='relative', help='Type of diversity metric to use for this experiment (ie. absolute, relative, original etc.)')
parser.add_argument('--pairwise_diversity_op', default='mean', help='the function to use for calculating diversity metric with regard to pairwise comparisons', type=str)   
parser.add_argument('--layerwise_diversity_op', default='w_mean', help='the function to use for calculating diversity metric with regard to layerwise comparisons', type=str)
parser.add_argument('--k', help='If using k-neighbors for metric calculation, how many neighbors', type=int, default=-1)
parser.add_argument('--k_strat', help='If using k-neigbhors for metric, what strategy should be used? (ie. closest, furthest, random, etc.)', type=str, default='closest')   

parser.add_argument('--continue_from_ckpt', help='continue training from ckpt', action='store_true', default=False)
parser.add_argument('--continue_at_epoch', help='continue with training a certain epoch', default=7)

args = parser.parse_args()

def run():

    torch.multiprocessing.freeze_support()

    stored_filters = {}
    
    experiment_name = args.experiment_name
    training_interval = args.training_interval
    fixed_conv = args.fixed_conv
    if args.rand_tech:
        name = args.rand_tech
    elif not args.rand_tech and not args.gram_schmidt and not args.ae:
        name = 'fitness'
    
    if args.gram_schmidt:
        name = 'gram-schmidt'
    
    if args.ae:
        name = 'ae_unsup'
        if training_interval < 1:
            print('please enter valid training interval as ae filters are in the shape num_runs, 1, filters')
            exit()
    if args.unique_id != "":
        name = 'current_' + name + "_" + args.unique_id
    
    filename = ''
    if not args.continue_from_ckpt:
        filename = 'output/' + experiment_name + '/solutions_over_time_{}.npy'.format(name)

        # get filters from numpy file
        np_load_old = partial(np.load)
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
        stored_filters = np.load(filename)
        np.load = np_load_old

        if args.unique_id != '':
            stored_filters = [stored_filters]

    # TODO: jdonovancs - the training for continuing will act like this is an initialization and will not 
    # take into account the # of epochs, lr decay, or any other factors unless I write that in here or helper
    else:
        stored_filters=[]
        count = 0
        path_to_pth = 'trained_models/trained/conv' + str(not args.fixed_conv) + '_e' + args.experiment_name + '_n' + name + "_r" + str(count) + 'g0.pth/perturbed-initializations/'
        while os.path.exists(path_to_pth) and len(os.listdir(path_to_pth)) > 0:
            path_to_ckpt = path_to_pth+os.listdir(path_to_pth)[0] + '/checkpoints/'
            for f in os.listdir(path_to_ckpt):
                if ('epoch=' + args.continue_at_epoch + '-') in f:
                    path_to_ckpt = path_to_ckpt + f
            stored_filters.append(helper.get_weights_from_ckpt(path_to_ckpt))
            count+=1
            path_to_pth = 'trained_models/trained/conv' + str(not args.fixed_conv) + '_e' + args.experiment_name + '_n' + name + "_r" + str(count) + 'g0.pth/perturbed-initializations/'


    helper.run(seed=False, rank=args.local_rank if args.devices > 0 else 0)


    # get loader for train and test images and classes
    data_module = helper.get_data_module(args.dataset, args.batch_size, args.num_workers)
    data_module.prepare_data()
    data_module.setup()

    epochs = args.epochs
    
    helper.config['bn'] = not args.no_bn
    helper.config['dataset'] = args.dataset.lower()
    helper.config['batch_size'] = args.batch_size
    helper.config['lr'] = args.lr
    helper.config['experiment_name'] = args.experiment_name
    helper.config['evo'] = not args.rand_tech and not args.gram_schmidt and not args.ae
    helper.config['experiment_type'] = 'training'
    helper.config['fixed_conv'] = fixed_conv == True
    helper.config['diversity_type'] = args.diversity_type
    helper.config['ae'] = args.ae
    helper.config['pairwise_diversity_op'] = args.pairwise_diversity_op
    helper.config['layerwise_diversity_op'] = args.layerwise_diversity_op
    helper.config['k'] = args.k
    helper.config['k_strat'] = args.k_strat
    helper.config['early_stopping'] = args.early_stopping
    helper.config['scheduler'] = args.use_scheduler
    helper.update_config()
    
    
    # run training and evaluation and record metrics in above variables
    # for each type of evolution ran
    inner_skip = args.inner_skip+1
    skip = args.skip
    stop_after = args.stop_after
    inner_stop_after = args.inner_stop_after if training_interval < 1 else len(stored_filters[0])
    
    # print(skip, stop_after, inner_skip, inner_stop_after, len(stored_filters), len(stored_filters[0]))

    for run_num in range(int(skip), min(skip+stop_after, len(stored_filters))):
        # run_num = np.where(stored_filters == filters_list)[0][0]
        # for each generation train the solution output at that generation
        inner_interval = (int((training_interval)*len(stored_filters[run_num])))
        if inner_interval == 0:
            inner_interval = 1
        print("inner interval", inner_interval, "inner_skip", inner_skip, "inner_stop_after", inner_stop_after)
        loop_range_start = int(inner_skip*inner_interval)-1
        loop_range_end = min(int(inner_skip*inner_interval*inner_stop_after), len(stored_filters[run_num])) 
        loop_range_interval = max (1, inner_interval)
        for n in range(loop_range_start, loop_range_end, loop_range_interval):
            if n < 0:
                continue
            # i = n-1
            # if we only want to train the solution from the final generation, continue
            # if (training_interval != 0 and i*1.0 not in [(len(stored_filters[run_num])/(1/training_interval)*j)-1 for j in range(1, min(args.stop_after, int(1/training_interval)+1))]) or (training_interval==0 and i not in range(skip, args.stop_after)):
            #     continue
            scaled = False
            if len(stored_filters[run_num][n]) > 6:
                scaled = True
            if args.diversity_type == "None":
                diversity = None
            else:
                diversity = {'type': args.diversity_type, 'pdop': args.pairwise_diversity_op, 'ldop': args.layerwise_diversity_op, 'k': args.k, 'k_strat': args.k_strat}

            # else train the network and collect the metrics
            helper.config['generation'] = n+1 if (not args.rand_tech and not args.gram_schmidt) else None
            helper.update_config()
            save_path = "trained_models/trained/conv{}_e{}_n{}_r{}_g{}.pth".format(not fixed_conv, experiment_name, name, run_num, n)
            print('Training and Evaluating: {} Gen: {} Run: {}'.format(name, n, run_num))
            record_progress = helper.train_network(data_module=data_module, filters=stored_filters[run_num][n], epochs=epochs, lr=args.lr, save_path=save_path, fixed_conv=fixed_conv, novelty_interval=int(args.novelty_interval), val_interval=int(args.val_accuracy_interval), diversity=diversity, scaled=scaled, devices=args.devices, save_interval=args.save_interval, bn=not args.no_bn, log_activations=args.log_activations, early_stopping=args.early_stopping, use_scheduler=args.use_scheduler)
            if n+1*loop_range_interval >= loop_range_end:
                continue
            helper.run(seed=False, rank=args.local_rank if args.local_rank > 0 else 0)
            helper.config['bn'] = not args.no_bn
            helper.config['dataset'] = args.dataset.lower()
            helper.config['batch_size'] = args.batch_size
            helper.config['lr'] = args.lr
            helper.config['experiment_name'] = args.experiment_name
            helper.config['evo'] = not args.rand_tech and not args.gram_schmidt and not args.ae
            helper.config['experiment_type'] = 'training'
            helper.config['fixed_conv'] = fixed_conv == True
            helper.config['diversity_type'] = args.diversity_type
            helper.config['ae'] = args.ae
            helper.config['pairwise_diversity_op'] = args.pairwise_diversity_op
            helper.config['layerwise_diversity_op'] = args.layerwise_diversity_op
            helper.config['k'] = args.k
            helper.config['k_strat'] = args.k_strat
            helper.config['early_stopping'] = args.early_stopping
            helper.config['scheduler'] = args.use_scheduler
            helper.update_config()
            # for c in classlist:
            #     classwise_accuracy_record[run_num][i][np.where(classlist==c)[0][0]] = record_accuracy[c]

    # with open('output/' + experiment_name + '/classwise_accuracy_{}over_time.pickle'.format(name_add), 'wb') as f:
    #     pickle.dump(classwise_accuracy_record,f)

if __name__ == '__main__':
    run()
