import wandb
import numpy as np

api = wandb.Api(timeout=30)
runs = api.runs(path="jdonovan/perturbed-initializations", filters={"config.experiment_name": {"$in": ["relative diversity", "absolute diversity", "cosine diversity"]}, "config.experiment_type": "evolution"})

rel_div_vals = []
abs_div_vals = []
cos_div_vals = []

for run in runs:
    print(run.id)
    div_vals_history = run.scan_history(keys=['best_individual_fitness'])
    div_vals = [row['best_individual_fitness'] for row in div_vals_history if not np.isnan(row['best_individual_fitness'])]
    
    if run.config['experiment_name'] == 'relative diversity':
        rel_div_vals.append(div_vals)
    if run.config['experiment_name'] == 'absolute diversity':
        abs_div_vals.append(div_vals)
    if run.config['experiment_name'] == 'cosine diversity':
        cos_div_vals.append(div_vals)


# manipulate data to be what I want it to be
rel_div_vals = [(np.array(vals)-min(vals))/(max(vals)-min(vals)) for vals in rel_div_vals]
mean_rel_div_vals = np.mean(rel_div_vals, axis=0)
std_rel_div_vals = np.std(rel_div_vals, axis=0)

abs_div_vals = [(np.array(vals)-min(vals))/(max(vals)-min(vals)) for vals in abs_div_vals]
mean_abs_div_vals = np.mean(abs_div_vals, axis=0)
std_abs_div_vals = np.std(abs_div_vals, axis=0)

np.errstate(invalid="ignore", divide='ignore')
cos_div_vals = [(np.array(vals)-min(vals)) for vals in cos_div_vals]
divisors = [max(d)-min(d) for d in cos_div_vals]
cos_div_vals = [np.array(vals)/(divisors[d]) for d, vals in enumerate(cos_div_vals) if divisors[d] != 0]
# cos_div_vals = [(np.array(vals)-min(vals))/(max(vals)-min(vals)) for vals in cos_div_vals]
mean_cos_div_vals = np.mean(cos_div_vals, axis=0)
std_cos_div_vals = np.std(cos_div_vals, axis=0)

import matplotlib.pyplot as plt
import helper_hpc as helper

plt.rcParams.update({'font.size': 22, "figure.figsize": (7, 6)})
input_data = [rel_div_vals, abs_div_vals, cos_div_vals]
helper.plot_mean_and_bootstrapped_ci_multiple([np.transpose(x) for x in input_data], "", ["relative", "absolute", "cosine"], "Gens", "Fitness (Regularized)", show=True)

# plt.figure()
# y = [i for i in range(len(mean_rel_div_vals))]
# plt.plot(y, mean_rel_div_vals, label='relative diversity')
# plt.fill_between(y, mean_rel_div_vals-std_rel_div_vals, mean_rel_div_vals+std_rel_div_vals, alpha=0.3)
# plt.plot(y, mean_abs_div_vals, label='absolute_diversity')
# plt.fill_between(y, mean_abs_div_vals-std_abs_div_vals, mean_abs_div_vals+std_abs_div_vals, alpha=0.3)
# plt.plot(y, mean_cos_div_vals, label='cosine diversity')
# plt.fill_between(y, mean_cos_div_vals-std_cos_div_vals, mean_cos_div_vals+std_cos_div_vals, alpha=0.3)
# plt.legend()
# plt.show()

