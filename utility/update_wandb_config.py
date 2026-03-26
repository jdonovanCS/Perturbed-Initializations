import wandb
import argparse

parser=argparse.ArgumentParser(description="Process some input files")
parser.add_argument('--run_ids', nargs='+', help='which run should be edited')
parser.add_argument('--param', help='parameter to change')
parser.add_argument('--param_type', help='typecast of param, such as int or str', type=str)
parser.add_argument('--value', help='value to change param to')
args = parser.parse_args()

for run in args.run_ids:
    run_id = run
    api = wandb.Api()
    run = api.run("jdonovan/perturbed-initializations/" + run_id)

    def confirm(prompt=None, resp=False):
        """prompts for yes or no response from the user. Returns True for yes and
        False for no.

        'resp' should be set to the default value assumed by the caller when
        user simply types ENTER.

        >>> confirm(prompt='Create Directory?', resp=True)
        Create Directory? [y]|n: 
        True
        >>> confirm(prompt='Create Directory?', resp=False)
        Create Directory? [n]|y: 
        False
        >>> confirm(prompt='Create Directory?', resp=False)
        Create Directory? [n]|y: y
        True

        """
        
        if prompt is None:
            prompt = 'Confirm'

        if resp:
            prompt = '%s [%s]|%s: ' % (prompt, 'y', 'n')
        else:
            prompt = '%s [%s]|%s: ' % (prompt, 'n', 'y')
            
        while True:
            ans = input(prompt)
            if not ans:
                return resp
            if ans not in ['y', 'Y', 'n', 'N']:
                print('please enter y or n.')
                continue
            if ans == 'y' or ans == 'Y':
                return True
            if ans == 'n' or ans == 'N':
                return False


    if not args.value:
        if confirm("Replace the value for this parameter with null / None?", True):
            value = None
    elif args.param_type == "int":
        value = int(args.value)
    elif args.param_type == "float":
        value = float(args.value)
    elif args.param_type == "bool":
        value = bool(int(args.value))
    else:
        value = str(args.value)

    if args.param.lower() == "state":
        my_project_name = "perturbed-initializations"
        my_id = run_id
        wandb.init(project=my_project_name, id=my_id, resume="must")
        wandb.finish()
    else:
        run.config[args.param] = value
    run.update()