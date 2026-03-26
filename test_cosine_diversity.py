import numpy as np
from functools import partial
import sys
# sys.path.append("..")
import helper_hpc as helper
import pytorch_lightning as pl
import torch
torch.set_printoptions(precision=32)
from net import Net



# relative diversity k10 prms
# relative diversity k10 furthest
# relative diversity k10 lmean
# relative diversity k10 prms lmean
# relative diversity k10

def old():
    e_file = 'D:/Learning/UVM/Research Projects/Novel-Feature-Detector/output/cosine diversity/solutions_over_time_fitness.npy'
    ind_files = ['D:/Learning/UVM/Research Projects/Novel-Feature-Detector/output/cosine diversity/solutions_over_time_current_fitness_4GoUc2o6r8mhbwgeJHhmbq.npy',
                'D:/Learning/UVM/Research Projects/Novel-Feature-Detector/output/cosine diversity/solutions_over_time_current_fitness_2F58aoXfzekUrApkozdLS3.npy',
                'D:/Learning/UVM/Research Projects/Novel-Feature-Detector/output/cosine diversity/solutions_over_time_current_fitness_MFrRiPopgx2XL8ayNcBimX.npy',
                'D:/Learning/UVM/Research Projects/Novel-Feature-Detector/output/cosine diversity/solutions_over_time_current_fitness_Ue4JSwBZnSPAqkmvumrvAV.npy',
                'D:/Learning/UVM/Research Projects/Novel-Feature-Detector/output/cosine diversity/solutions_over_time_current_fitness_M8F2Q3C5rsZy5xW93h4MMn.npy',
                'D:/Learning/UVM/Research Projects/Novel-Feature-Detector/output/cosine diversity/solutions_over_time_current_fitness_jXYHANxKdYpecsXieRcAzs.npy',
                'D:/Learning/UVM/Research Projects/Novel-Feature-Detector/output/cosine diversity/solutions_over_time_current_fitness_664cCq9L8wwiS8dGq8Ehz3.npy',
                'D:/Learning/UVM/Research Projects/Novel-Feature-Detector/output/cosine diversity/solutions_over_time_current_fitness_SZWsfsVoQ2BsfJYUs64oc8.npy',
                'D:/Learning/UVM/Research Projects/Novel-Feature-Detector/output/cosine diversity/solutions_over_time_current_fitness_i2cQVkkjhBw2m2FD9cpeGh.npy',
                'D:/Learning/UVM/Research Projects/Novel-Feature-Detector/output/cosine diversity/solutions_over_time_current_fitness_d8XLvqiRaztbiQvorcD2mt.npy']

    np_load_old = partial(np.load)
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    e_filters = np.load(e_file)
    i_filters = [np.load(f) for f in ind_files]
    np.load = np_load_old

    print((e_filters.dtype))
    print((e_filters[0].dtype))
    print(len(e_filters[0][0]))
    print((e_filters[0][0][0].dtype))
    print((e_filters[0][0][0][0].dtype))
    print((e_filters[0][0][0][0][0].dtype))
    print((e_filters[0][0][0][0][0][0].dtype))
    print((e_filters[0][0][0][0][0][0][0].dtype))
    print()


    print((type(e_filters)))
    print((type(e_filters[0])))
    print(len(e_filters[0][0]))
    print((type(e_filters[0][0][0])))
    print((type(e_filters[0][0][0][0])))
    print((type(e_filters[0][0][0][0][0])))
    print((type(e_filters[0][0][0][0][0][0])))
    print((type(e_filters[0][0][0][0][0][0][0])))
    print()

    print((e_filters.shape))
    print((e_filters[0].shape))
    print(len(e_filters[0][0]))
    print((e_filters[0][0][0].shape))
    print((e_filters[0][0][0][0].shape))
    print((e_filters[0][0][0][0][0].shape))
    print((e_filters[0][0][0][0][0][0].shape))
    print((e_filters[0][0][0][0][0][0][0].shape))
    print()

    import torch
    torch.set_printoptions(precision=32)

    print((e_filters[0][0][0][0][0]))
    print((e_filters[0][0][0][0][1]))
    print()

    # For each run
    total_diffs=0
    indices = []
    for i in range(len(i_filters)):
        # print(helper.cosine_dist(np.array(e_filters[i][49][0][0][0]).flatten(), np.array(e_filters[i][49][0][0][1]).flatten()))
        # print(helper.cosine_dist(np.array(e_filters[i][6][0][0][0]).flatten(), np.array(e_filters[i][6][0][0][1]).flatten()))
        # For each generation
        num_diffs = 0
        num_diffs_by_val = 0
        total = 0
        for j in range(0, len(e_filters[i]), len(e_filters[i])):
            
            # For each layer
            # if total_diffs != 0:
            #     print('total', total_diffs)
            total_diffs=0
            for k in range(len(e_filters[i][j])):
                # print(torch.std(i_filters[i][j][k]))
                # print(torch.mean(i_filters[i][j][k]))
                if torch.sum(i_filters[i][j][k] == i_filters[i][49][k]) != 0:
                    layer_diff = int(torch.sum(i_filters[i][j][k] != i_filters[i][49][k]))/9
                    num_diffs += layer_diff
                    print('layer_diff', layer_diff)
                else:
                    total_diffs += len(i_filters[i][j][k].flatten())/9

                # For each 3D filter
                for l in range(len(e_filters[i][j][k])):
                    
                    # for each 3x3 filter
                    for m in range(len(e_filters[i][j][k][l])):

                        if torch.sum(torch.abs(i_filters[i][49][k][l][m]) > (1/np.sqrt(len(i_filters[i][j][k][l])))+.1) > 0:
                            num_diffs_by_val += 1
                            indices.append((i, 49, k, l, m))
                        total += 1
        #                 diff=diff+1 if torch.sum(torch.eq(i_filters[i][j][k][l], i_filters[i][49][k][l])) != len(i_filters[i][j][k][l].flatten()) else diff
        #                 count+=1

        #                 for n in range(len(e_filters[i][j][k][l][m])):
        #                     diff2=diff2+1 if torch.sum(torch.eq(i_filters[i][j][k][l][m], i_filters[i][49][k][l][m])) != len(i_filters[i][j][k][l][m].flatten()) else diff2
        #                     count2+=1
        print('by comp:', num_diffs)
        print('by val:', num_diffs_by_val)
        # print('total:', total_diffs)
        # print('innert total', total)


        # # helper.run(seed=False)
        # data_module = helper.get_data_module("random", batch_size=64, workers=0)
        # data_module.prepare_data()
        # data_module.setup()
        # trainer = pl.Trainer(accelerator="auto", limit_val_batches=1)
        # trainer2 = pl.Trainer(accelerator="auto", limit_val_batches=1)
        # classnames = list(data_module.dataset_test.classes)
        # net = helper.Net(num_classes=len(classnames), classnames=classnames, diversity={"type": 'cosine', "pdop": 'mean', "ldop":'w_mean', 'k': -1, 'k_strat': 'closest'})
        # net2 = helper.Net(num_classes=len(classnames), classnames=classnames, diversity={"type": 'cosine', "pdop": 'mean', "ldop":'w_mean', 'k': -1, 'k_strat': 'closest'})
        # net.set_filters(i_filters[i][10])
        # net2.set_filters(i_filters[i][49])
        # trainer.validate(net, dataloaders=data_module.val_dataloader(), verbose=False)
        # trainer2.validate(net2, dataloaders=data_module.val_dataloader(), verbose=False)
        # print(net.avg_novelty, net2.avg_novelty)
            

    print(indices)
    # for i in range(len(ind_files)):
    #     for j in range(len(e_filters[j][0])):
    #         assert(torch.sum(torch.eq(e_filters[i][49][j], i_filters[i][49][j])) == )
                # print('match: ' + ind_files[i] + '== e_filters[' + str(j) + ']')
                    



    # print(sys.getsizeof(e_filters))

# old()
    
def func2():
    e_file = 'D:/Learning/UVM/Research Projects/Novel-Feature-Detector/output/mutate 750 weighted .75 .04 .05 .05 .05 .06/solutions_over_time_mutate-only.npy'
    np_load_old = partial(np.load)
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    e_filters = np.load(e_file)
    np.load = np_load_old

    glob_layer_diffs = []
    indices = []
    non_mutated = []
    mutated = []
    mutated_layer = []
    non_mutated_layer = []
    # for i in [5, 347, 13, 22, 138, 188, 94, 82, 48, 130, 305, 424, 471, 393, 430, 455, 351, 145, 419]: 
    for i in range(min(len(e_filters), 1000)):
        # For each network
        num_diffs_by_val = 0
        total = 0
        layer_diffs = []
        total_layer = []
        for j in range(0, len(e_filters[i])):
            
            # For each layer
            total_diffs=0
            for k in range(len(e_filters[i][j])):
                mutated_layer.append([])
                non_mutated_layer.append([])
                layer_diffs.append(0)
                total_layer.append(0)

                # For each 3D filter
                for l in range(len(e_filters[i][j][k])):
                    
                    # for each 3x3 filter
                    for m in range(len(e_filters[i][j][k][l])):

                        if torch.sum(torch.abs(e_filters[i][j][k][l][m]) > (1/np.sqrt(len(e_filters[i][j][k][l])))+.1) > 0:
                            layer_diffs[k]+=1
                            num_diffs_by_val += 1
                            indices.append((k, l, m))
                            mutated.append(abs(e_filters[i][j][k][l][m]).mean())
                            mutated_layer[k].append(abs(e_filters[i][j][k][l][m]).mean())
                        else:
                            non_mutated.append(abs(e_filters[i][j][k][l][m]).mean())
                            non_mutated_layer[k].append(abs(e_filters[i][j][k][l][m]).mean())
                        total += 1
                        total_layer[k] += 1

        glob_layer_diffs.append(np.array(layer_diffs))
        print(np.mean(np.array(glob_layer_diffs), axis=0))
        print('run num', i)
        print('layer diffs', layer_diffs)
        print('total layer filters', total_layer)
        print('num diffs by val', num_diffs_by_val)
        print('total filters', total)
        print()
        print()
    

    glob_layer_diffs = np.array(glob_layer_diffs)
    mean_glob_layer_diffs = np.mean(glob_layer_diffs, axis=1)

    print('mean global layer diffs', mean_glob_layer_diffs)

    print('mutated magnitude mean', np.array(mutated).mean())
    print('non mutated magnitude mean', np.array(non_mutated).mean())
    print()
    print('mutated magnitude std', np.array(mutated).std())
    print('non mutated magnitude std', np.array(non_mutated).std())
    print()
    print()

    mutated_layer_means = []
    non_mutated_layer_means = []
    
    mutated_layer_std = []
    non_mutated_layer_std = []

    for k in range(len(e_filters[0][0])):
        mutated_layer_means.append(np.array(mutated_layer[k]).mean())
        mutated_layer_std.append(np.array(mutated_layer[k]).std())
        non_mutated_layer_means.append(np.array(non_mutated_layer[k]).mean())
        non_mutated_layer_std.append(np.array(non_mutated_layer[k]).std())

    print('mutated layer magnitude means', mutated_layer_means)
    print('non mutated layer magnitude means', non_mutated_layer_means)
    print()
    print('mutated layer magnitude std', non_mutated_layer_std)
    print('non mutated layer magnitude std', non_mutated_layer_std)
    print()

    
    return

    n = Net.load_from_checkpoint('D:/Learning/UVM/Research Projects/Novel-Feature-Detector/trained_models/trained/convTrue_emutate weighted 750_nmutate-only_r0_g0.pth/perturbed-initializations/0z402yzx/checkpoints/epoch=255-step=160000.ckpt')

    num_diffs_by_val = 0
    layer_diffs = []
    t_filters = n.get_filters()
    non_mutated = []
    mutated = []
    mutated_layer = []
    non_mutated_layer = []
    for k in range(len(t_filters)):
        mutated_layer.append([])
        non_mutated_layer.append([])
        layer_diffs.append(0)

        for l in range(len(t_filters[k])):

            for m in range(len(t_filters[k][l])):

                if torch.sum(torch.abs(t_filters[k][l][m]) > .2) > 0:
                    layer_diffs[k]+=1
                    num_diffs_by_val += 1
                
                if (k, l, m) in indices:
                    mutated.append(abs(t_filters[k][l][m]).mean())
                    mutated_layer[k].append(abs(t_filters[k][l][m]).mean())
                else:
                    non_mutated.append(abs(t_filters[k][l][m]).mean())
                    non_mutated_layer[k].append(abs(t_filters[k][l][m]).mean())
    
    print('layer diffs', layer_diffs)
    print('num diffs by val', num_diffs_by_val)
    print('mutated magnitude mean', np.array(mutated).mean())
    print('non mutated magnitude mean', np.array(non_mutated).mean())
    print()
    print('mutated magnitude std', np.array(mutated).std())
    print('non mutated magnitude std', np.array(non_mutated).std())
    print()
    print()

    mutated_layer_means = []
    non_mutated_layer_means = []
    
    mutated_layer_std = []
    non_mutated_layer_std = []

    for k in range(len(t_filters)):
        mutated_layer_means.append(np.array(mutated_layer[k]).mean())
        mutated_layer_std.append(np.array(mutated_layer[k]).std())
        non_mutated_layer_means.append(np.array(non_mutated_layer[k]).mean())
        non_mutated_layer_std.append(np.array(non_mutated_layer[k]).std())

    print('mutated layer magnitude means', mutated_layer_means)
    print('non mutated layer magnitude means', non_mutated_layer_means)
    print()
    print('mutated layer magnitude std', non_mutated_layer_std)
    print('non mutated layer magnitude std', non_mutated_layer_std)
    print()



func2()

def create_pruned_networks():

    e_file = 'C:/Users/Jordan/Learning/UVM/Research/novel-feature-detector/output/mutate weighted 750/solutions_over_time_mutate-only.npy'
    np_load_old = partial(np.load)
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    e_filters = np.load(e_file)
    np.load = np_load_old

    
    for i in range(len(e_filters)):
        # For each network
        
        for j in range(0, len(e_filters[i])):
            
            # For each layer
            for k in range(len(e_filters[i][j])):

                # For each 3D filter
                for l in range(len(e_filters[i][j][k])):
                    
                    # for each 3x3 filter
                    for m in range(len(e_filters[i][j][k][l])):

                        if torch.sum(torch.abs(e_filters[i][j][k][l][m]) > (1/np.sqrt(len(e_filters[i][j][k][l])))+.1) > 0:
                            continue
                        else:
                            e_filters[i][j][k][l][m] = torch.zeros(e_filters[i][j][k][l][m].size())

    e_file = e_file[:-4] + '_pruned.npy'
    np.save(e_file, e_filters)

# create_pruned_networks()
# func2()



