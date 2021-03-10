import argparse
import datetime
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt

from rl_playground import run_tests
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'path',
        type=str,
        help=(
            'Arguments JSON file to load, filename with '
            'timestamp right before extension, or timestamp.'
        ),
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Model file to load. Has priority over `path`.',
    )
    return parser.parse_args()


def _script_start_from_arg(arg):
    file_extension_start_index = arg.rfind('.')
    file_extension_start_rindex = len(arg) - file_extension_start_index
    if file_extension_start_rindex == 7:
        # We don't have a file name but only the script start time
        return arg[-26:]
    return arg[-file_extension_start_rindex-26:-file_extension_start_rindex]


def get_prev_args(test_args):
    args_path = Path(test_args.path)
    if not args_path.exists() or not args_path.suffix.lower() == '.json':
        prev_script_start = _script_start_from_arg(test_args.path)
        args_path = Path(f'args_{prev_script_start}.json')

    with open(args_path, 'r') as f:
        args = json.load(f)
    args = argparse.Namespace(**args)
    if not hasattr(args, 'script_start'):
        args.script_start = prev_script_path


def model_fname_from_args(test_args, args):
    if test_args.model_path is not None:
        fname = Path(test_args.model_path)
        assert fname.exists(), f"checkpoint file {fname} does not exist"
        return fname

    fname = Path(f'sdc_model_{args.model_class.lower()}_'
                 f'{args.policy_class.lower()}_'
                 f'{args.script_start}.zip')
    if not fname.exists():
        learning_rate = utils.compute_learning_rate(args)
        fname = Path(f'sdc_model_{args.model_class.lower()}_'
                     f'{args.policy_class.lower()}_{learning_rate}_'
                     f'{args.script_start}.zip')
    assert fname.exists(), (
        "checkpoint file could not be determined from arguments; "
        "please use the `--model_path` argument.")
    return fname


def main():
    script_start = str(datetime.datetime.now()
                       ).replace(':', '-').replace(' ', 'T')
    test_args = parse_args()
    test_args.script_start = script_start

    args = get_prev_args(test_args)

    # Only save when we were able to load the previous arguments
    args_path = Path(f'test_args_{script_start}.json')
    with open(args_path, 'w') as f:
        json.dump(vars(test_args), f, indent=4)

    utils.setup(args.use_sb3, args.debug_nans)

    seed = args.seed
    eval_seed = seed
    if eval_seed is not None:
        eval_seed += args.num_envs

    policy_class = utils.get_policy_class(args.policy_class, args.model_class)
    utils.check_num_envs(args, policy_class)

    fname = model_fname_from_args(test_args, args)

    # ---------------- TESTING STARTS HERE ----------------

    fig_path = Path(f'test_results_{script_start}.pdf')
    run_tests(fname, args, seed=eval_seed, fig_path=fig_path)


if __name__ == '__main__':
    main()
