import argparse
from pathlib import Path
import subprocess

import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--steps',
        type=float,
        default=10000,
        help='Number of action steps to take in the environment.',
    )

    parser.add_argument(
        '--M',
        type=int,
        default=3,
        help=(
            '"Difficulty" of the problem '
            '(proportionally relates to nth-order differential equation)'
            '(choose M = 3 or M = 5 for comparing MIN-preconditioner)'
        ),
    )
    parser.add_argument(
        '--dt',
        type=float,
        default=1.0,
        help=(
            'Size of time step.'
        ),
    )
    parser.add_argument(
        '--restol',
        type=float,
        default=1E-10,
        help=(
            'Residual tolerance; '
            'when residual is below this, stop iterating.'
            'Lower means better solution but more iterations.'
        ),
    )

    parser.add_argument(
        '--lambda_real_interval',
        type=int,
        nargs=2,
        default=[-100, 0],
        help='Interval to sample the real part of lambda from.',
    )
    parser.add_argument(
        '--lambda_imag_interval',
        type=int,
        nargs=2,
        default=[0, 0],
        help='Interval to sample the imaginary part of lambda from.',
    )
    parser.add_argument(
        '--lambda_real_interpolation_interval',
        type=int,
        nargs=2,
        default=None,
        help=(
            'Interval of number of episodes to resize the real interval of '
            'lambda with (by decreasing the lower limit). '
            'By default, do not resize.'
        ),
    )

    parser.add_argument(
        '--envname',
        type=str,
        default='sdc-v0',
        choices=['sdc-v0', 'sdc-v1'],
        help=(
            'Gym environment to use;\n    sdc-v0 – SDC with a full iteration '
            'per step (no intermediate observations),\n    sdc-v1 – SDC with '
            'a single iteration per step (with intermediate observations).'
        ),
    )
    parser.add_argument(
        '--norm_obs',
        type=utils.parse_bool,
        default=False,
        help='Whether to normalize the observations during training.',
    )

    parser.add_argument(
        '--learning_rate',
        type=float,
        # default for PPO2 in stable-baselines
        default=25E-05,
        help='Learning rate/step size of the model.',
    )
    parser.add_argument(
        '--rescale_lr',
        type=utils.parse_bool,
        default=True,
        help=(
            'Whether to rescale the learning rate by the number'
            'of environments.'
        ),
    )
    parser.add_argument(
        '--activation_fn',
        type=str,
        default=None,
        help=(
            'Policy activation function to use. '
            "Defaults to the policy's default."
        ),
    )

    parser.add_argument(
        '--model_class',
        type=str,
        default='PPO2',
        help='Class of model to instantiate.',
    )
    parser.add_argument(
        '--model_kwargs',
        type=utils.parse_dict,
        default={},
        help=(
            'Keyword arguments for model creation. '
            'See the documentation for details. '
            'The other arguments `learning_rate`, `policy_kwargs` and `seed` '
            'overwrite the values given here. '
            'Example for PPO: '
            '`--model_kwargs \'{"gamma": 0.999}\'`'
        ),
    )
    parser.add_argument(
        '--policy_class',
        type=str,
        default='MlpPolicy',
        help='Class of model policy.',
    )
    parser.add_argument(
        '--policy_kwargs',
        type=utils.parse_dict,
        default={},
        help=(
            'Keyword arguments for policy creation. '
            'See the documentation for details.'
            'Example for MlpLstmPolicy: '
            '`--policy_kwargs \'{"net_arch": [128, 128, "lstm"]}\'`'
        ),
    )

    parser.add_argument(
        '--norm_factor',
        type=float,
        default=1.0,
        help=(
            'How to scale residual norms ',
            '(if `--reward_iteration_only True`).'
        ),
    )
    parser.add_argument(
        '--residual_weight',
        type=float,
        default=0.5,
        help=(
            'How to scale the residual reward '
            '(if `--reward_iteration_only True`).'
        ),
    )
    parser.add_argument(
        '--step_penalty',
        type=float,
        default=0.1,
        help='Base value to penalize each timestep.',
    )
    parser.add_argument(
        '--reward_iteration_only',
        type=utils.parse_bool,
        default=True,
        help=(
            'How to reward the agent. '
            'Set to `False` to reward based on the residual.'
        ),
    )
    parser.add_argument(
        '--collect_states',
        type=utils.parse_bool,
        default=False,
        help='Whether to collect all previous states as input.',
    )

    parser.add_argument(
        '--tests',
        type=float,
        default=5000,
        help='Number of test runs for each preconditioning method.',
    )
    parser.add_argument(
        '--num_envs',
        type=int,
        default=4,
        help='How many environments to use in parallel.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help=(
            'Base seed for seeding the environments. For multiple '
            'environments, all will have different seeds based on this one.'
        ),
    )
    parser.add_argument(
        '--save_freq',
        type=int,
        default=0,
        help=(
            'How often to save checkpoints of the model during training. '
            'If this is 0, do not checkpoint.'
        ),
    )
    parser.add_argument(
        '--eval_freq',
        type=int,
        default=0,
        help=(
            'How often to evaluate the model during training, storing the '
            'best performing one. If this is 0, do not evaluate.'
        ),
    )

    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help=(
            'Model checkpoint to load (a ZIP file). '
            'Will reconstruct the model from the checkpoint, '
            'ignoring model arguments (unless `--change_model True`).'
        ),
    )
    parser.add_argument(
        '--change_model',
        type=utils.parse_bool,
        default=False,
        help=(
            "Whether to overwrite a loaded model's parameters with the "
            'model arguments. (`policy_kwargs` cannot be changed.)'
        ),
    )

    parser.add_argument(
        '--use_sb3',
        type=utils.parse_bool,
        default=utils.has_sb3(),
        help=(
            'Whether to use stable-baselines3. '
            'Defaults to `True` if it is available, otherwise `False`.'
        ),
    )
    parser.add_argument(
        '--debug_nans',
        type=utils.parse_bool,
        default=False,
        help='Whether to enable NaN debugging.',
    )

    args = parser.parse_args()

    git_dir = Path(__file__).parent.parent / '.git'
    git_hash = subprocess.run(['git', '--git-dir', git_dir,
                               'rev-parse', 'HEAD'],
                              stdout=subprocess.PIPE).stdout
    args.git_hash = git_hash.decode().rstrip()
    return args
