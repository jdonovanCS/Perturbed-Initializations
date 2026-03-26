import wandb

# parser=argparse.ArgumentParser(description="Process some input files")
# parser.add_argument('--run_id', help='which run should be edited')
# parser.add_argument('--param', help='parameter to change')
# parser.add_argument('--param_type', help='typecast of param, such as int or str', type=str)
# parser.add_argument('--value', help='value to change param to')
# args = parser.parse_args()

# run_id = args.run_id
api = wandb.Api()
runs = api.runs(path="jdonovan/perturbed-initializations", filters={"config.experiment_name": {"$in": ["relative diversity", "absolute diversity", "cosine diversity"]}, "config.experiment_type": "evolution"})#"config.fixed_conv": False, "config.dataset": "cifar10"})#, "config.State": "Crashed", "config.evo": 0, "config.fixed_conv": 0})
# runs = api.runs('jdonovan/perturbed-initializations', filters={'created_at': {'$lt': '2023-06-20T20'}})

# data = []
# with open('artifacts/run-um0c4jr0-history-v0/0000.csv', newline='') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
#     for row in spamreader:
#         data.append(row)
# print(data)
# table = wandb.Table(columns = data[0], data=data[1:])

# run = api.run(path='jdonovan/perturbed-initializations/3doqm1v1')
# for artifact in run.logged_artifacts():
#     if artifact.type == "model":
#         artifact.delete(delete_aliases=True)
#     if artifact.type == "wandb-history":
#         print(dir(artifact))
#         print(dir(artifact._files))
#         print((artifact.manifest.to_manifest_json()))
#         artifact.add(table, "testing_update")
# artifact = api.artifact('jdonovan/perturbed-initializations/run-3doqm1v1-history:v0', type='wandb-history')
# artifact.delete()

print("{} runs found".format(len(runs)))
for run in runs:
    print(run.id)
    # for artifact in run.logged_artifacts():
    #     if artifact.type == "model":
    #         artifact.delete(delete_aliases=True)
    # print(run.State)
    # if run.State == 'crashed' or run.State == 'failed':
    #     print('deleting', run.id)
    #     run.delete()
    # # print([type(row['val_novelty']) for row in history])
    # values = [row['val_novelty'] for row in history]
    # if np.mean(values) < .2:
    #     values = np.array(values)*6
    # values2 = [row['val_novelty_epoch'] for row in history]
    # if np.mean(values2) < .2:
    #     values2 = np.array(values)*6
    # run.summary["val_novelty_corrected"] = values
    # run.summary["val_novelty_epoch_corrected"] = values2
# if not args.value:
#     if confirm("Replace the value for this parameter with null / None?", True):
#         value = None
# elif args.param_type == "int":
#     value = int(args.value)
# elif args.param_type == "float":
#     value = float(args.value)
# elif args.param_type == "bool":
#     value = bool(int(args.value))
# else:
#     value = str(args.value)
    # run.summary.update()