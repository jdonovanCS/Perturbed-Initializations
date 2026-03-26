# import any dependencies
import numpy as np
import helper_hpc as helper
import os
from model import Model
import torch
from ae_net import AE

def run():
    torch.multiprocessing.freeze_support()
    helper.run(seed=False)
    
    # create empty holder
    solution_results= {}
    solution_results['ae_unsup'] = np.array([[Model() for i in range(1)]for j in range(5)], dtype=object)

    # loop through the models created
    for i in range(5):

        # for each grab the convolutional filter weights from them
        for j in range(len(os.listdir('trained_models/trained/ae_ecifar100_ae_1_1000_normal_r{}.pth/perturbed-initializations/'.format(i)))):
            if os.path.exists(os.path.join('trained_models/trained/ae_ecifar100_ae_1_1000_normal_r{}.pth'.format(i), 'perturbed-initializations/', os.listdir('trained_models/trained/ae_ecifar100_ae_1_1000_normal_r{}.pth/perturbed-initializations/'.format(i))[j], 'checkpoints/epoch=0-step=625.ckpt')):
                path = os.path.join('trained_models/trained/ae_ecifar100_ae_1_1000_normal_r{}.pth'.format(i), 'perturbed-initializations/', os.listdir('trained_models/trained/ae_ecifar100_ae_1_1000_normal_r{}.pth/perturbed-initializations/'.format(i))[j], 'checkpoints/epoch=0-step=625.ckpt')
        m = AE.load_from_checkpoint(path)

        with torch.no_grad():
            filters = m.get_filters(numpy=True)

            # add these to a numpy array with the same structure as the ones created by evolution
            solution_results['ae_unsup'][i] = [filters]

    # save this np array to file using npy.save (make sure the experiment name / save location is correct)
    if not os.path.isdir('output/cifar100_ae_1_1000_normal'):
        os.mkdir('output/cifar100_ae_1_1000_normal')
    with open('output/cifar100_ae_1_1000_normal/solutions_over_time_{}.npy'.format('ae_unsup'), 'wb') as f:
        np.save(f, solution_results['ae_unsup'])

if __name__ == '__main__':
    run()