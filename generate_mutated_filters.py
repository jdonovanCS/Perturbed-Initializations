from tqdm import tqdm
import helper_hpc as helper
import os
import numpy as np
import torch
import argparse
from model import Model
import copy
import glob
from functools import partial


parser=argparse.ArgumentParser(description="Process some input files")
parser.add_argument('--experiment_name', help='experiment name for saving and data related to filters generated', default='')
parser.add_argument('--input_experiment_name', help='name of input experiment filters to perturb', default=None)
# parser.add_argument('--population_size', help='number of filters to generate', type=int, default=50)
# parser.add_argument('--technique', help='uniform, normal, gram-schmidt, or mutate-only technique', type=str, default='uniform')
# parser.add_argument('--network', help="specify network to generate filters for (vgg16, conv6, etc.)", type=str, default='conv6')
parser.add_argument('--mr', help="mutation rate to use if using mutation-only", default=1.0, type=float)
parser.add_argument('--broad_mut', help="use broad muation", default=False, action="store_true")
parser.add_argument('--num_mutations', help='how many times to run the mutation function on the filters', default=500, type=int)
parser.add_argument('--gain', help='optional scaling factor for some of the technqiues', default=1.0, type=float)
parser.add_argument('--weighted_mut', help="would we like for the mutation function to weight its selection based on number of filters in each layer", default=True, action='store_true')
parser.add_argument('--weights_for_mut', help='specify weights for each layer during mutation', nargs='+', default=None, type=float)
parser.add_argument('--ensure_mut_at_each_layer', help='use this option to ensure that at least 1 filter at each layer is perturbed', default=False, action='store_true')
# parser.add_argument('--batch_size', help='Number of images to use for novelty metric, only 1 batch used', default=64, type=int)
# parser.add_argument('--dataset', help='which dataset should be used for novelty metric, choices are: random, cifar-10', default='random')
args = parser.parse_args()

if args.input_experiment_name == None:
    print('must provide input file for perturbing')
    exit()
if args.weights_for_mut and (sum(args.weights_for_mut) < 0.99999999 or sum(args.weights_for_mut) > 1.000000001 or len(args.weights_for_mut) not in [6,13]):
    print('weights must add to 1.0. Current sum is: ', sum(args.weights_for_mut))
    exit()

def run():
    torch.multiprocessing.freeze_support()
    helper.run(seed=False)
    
    # global trainloader
    # if args.dataset.lower() != 'cifar-10' and args.dataset.lower() != 'cifar10':
    #     random_image_paths = helper.create_random_images(64)
    #     data_module = helper.load_random_images(random_image_paths)
    # else:
    #     data_module = helper.get_data_module(args.dataset.lower(), batch_size=args.batch_size)

    # data_module.prepare_data(data_dir="data/")
    # data_module.setup()
    # classnames = list(data_module.dataset_test.classes)

    # data_iterator = iter(data_module.train_dataloader())
    # net_input = next(data_iterator)
    
    filename = glob.glob('output/' + args.input_experiment_name + '*/solutions_over_time*.npy')[0]

    # get filters from numpy file
    np_load_old = partial(np.load)
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    stored_filters = np.load(filename)
    np.load = np_load_old
    
    experiment_name = args.experiment_name   
    population_size = len(stored_filters) 
    population = []
    helper.config['experiment_name'] = experiment_name

    mutated_filter_indices = []
    

    for i, net_filters in enumerate(tqdm(stored_filters)): #while len(population) < population_size:
        # print(net_filters.shape)
        model = Model()
        if len(net_filters[0]) > 6: #(vgg16)
            net = helper.BigNet()
        elif len(net_filters[0]) == 6: #(conv6)
            net = helper.Net()
        model.filters = copy.deepcopy(net_filters[0])
        mutated_filter_indices.append([])
        for k in range(args.num_mutations):
            if args.ensure_mut_at_each_layer and k<len(model.filters):
                mutated_filter_indices[i].append(helper.choose_mutate_index_from_layer(model.filters, k))
            else:
                mutated_filter_indices[i].append(helper.choose_mutate_index(model.filters, args.weighted_mut, args.weights_for_mut))
            model.filters = helper.mutate(model.filters, args.broad_mut, args.mr, mutated_filter_indices[i][k])
        net.set_filters(copy.deepcopy(model.filters))
        model.filters = net.get_filters()
        population.append(model)
    # helper.wandb.log({'gen': 0, 'individual': i, 'fitness': model.fitness})
        
    # sols = [p.filters for p in population]
    sols = [p.filters for p in population]
    solutions = np.array([[Model() for i in range(1)]for j in range(population_size)], dtype=object)
    
    for i in range(population_size):
        solutions[i][0] = sols[i]
    # solutions = solutions[0]
    sol_dict = {"mutate-only": solutions}
    # fitnesses = [p.fitness for p in population]

    if not os.path.isdir('output/' + experiment_name):
        os.mkdir('output/' + experiment_name)
    # with open('output/' + experiment_name + '/random_gen_solutions.pickle', 'wb') as f:
    #     pickle.dump(sol_dict, f)
    for k,v in sol_dict.items():
        with open('output/' + experiment_name + '/solutions_over_time_{}.npy'.format(k), 'wb') as f:
            np.save(f, v[::])

    for k,v in sol_dict.items():
        with open('output/' + experiment_name + '/mutated_filter_indices_{}.npy'.format(k), 'wb') as f:
            np.save(f, mutated_filter_indices)

    # print(mutated_filter_indices)
    # with open('output/' + experiment_name + '/random_gen_fitnesses.txt', 'a+') as f:
    #     f.write(str(fitnesses))

    # fitnesses = np.array([fitnesses])
    # cut_off_beginning = 0
    # helper.plot_mean_and_bootstrapped_ci_multiple(input_data=[np.transpose(x)[cut_off_beginning:] for x in fitnesses], name=[i for i in range(len(fitnesses))], x_label="Generation", y_label="Fitness", compute_CI=True)



if __name__ == '__main__':

    run()

