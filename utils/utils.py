from pathlib import Path

import gym
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
import stable_baselines

PPO2_DEFAULT_NUM_MINIBATCHES = 4


def get_model_class(model_class_str):
    """Return a model class according to `model_class_str`."""
    try:
        model_class = getattr(stable_baselines, model_class_str)
    except AttributeError:
        raise AttributeError(
            f"could not find model class '{model_class_str}' "
            f'in module `stable_baselines`'
        )
    assert issubclass(model_class, stable_baselines.common.BaseRLModel), \
        ('model class must be a subclass of '
         '`stable_baselines.common.BaseRLModel`')
    return model_class


def get_policy_class(policy_class_str, model_class_str):
    """Return a policy class according to `policy_class_str`.
    `model_class_str` is the model to create the policy for.
    """
    if model_class_str.upper() == 'DDPG':
        policy_class_module = stable_baselines.ddpg.policies
    elif model_class_str.upper() == 'DQN':
        policy_class_module = stable_baselines.deepq.policies
    else:
        policy_class_module = stable_baselines.common.policies

    try:
        policy_class = getattr(
            policy_class_module,
            policy_class_str,
        )
    except AttributeError:
        try:
            policy_class = globals()[policy_class_str]
        except KeyError:
            raise AttributeError(
                f"could not find policy class '{policy_class_str}' "
                f'in module `stable_baselines.common.policies` '
                f'or in this module'
            )
    assert issubclass(
        policy_class,
        stable_baselines.common.policies.BasePolicy,
    ), ('policy class must be a subclass of '
        '`stable_baselines.common.policies.BasePolicy`')
    return policy_class


def maybe_fix_ntests(ntests_given, num_test_envs):
    """Return `ntests_given` approximately scaled to a vectorized environment
    with `num_test_envs` parallel environments.

    Print a warning if the rescaling does not result in the same number of
    test runs.
    """
    # Amount of test loops to run
    ntests = ntests_given // num_test_envs * num_test_envs
    if ntests != ntests_given:
        print(f'Warning: `ntests` set to {ntests} ({ntests_given} was given).')
    return ntests


def find_free_path(format_path):
    """Return a path given by `format_path` into which a single incrementing
    number is interpolated until a non-existing path is found.
    """
    i = 0
    path = Path(format_path.format(i))
    while path.exists():
        i += 1
        path = Path(format_path.format(i))
    return path


def make_env(
        args,
        num_envs=None,
        include_norm=False,
        norm_obs=False,
        norm_reward=True,
        **kwargs,
):
    """Return a vectorized environment containing `num_envs` or `args.num_envs`
    environments (depending on whether `num_envs is None`).

    `args`, the command line arguments, specify several values. See `kwargs`
    for a more detailed explanation on their interaction.
    `include_norm` specifies whether the environment is wrapped in a
    normalizing environment.
    `norm_obs` and `norm_reward` indicate whether the observations or
    rewards are normalized (only revelant if `include_norm is True`).
    `kwargs` are passed directly to the environment creation function. Any
    value given via `kwargs` has priority over the one given by `args`.
    """
    if num_envs is None:
        num_envs = args.num_envs

    # `kwargs` given via `args`
    args_kwargs = {}
    for arg in [
            'M',
            'dt',
            'restol',
            'norm_factor',
            'residual_weight',
            'step_penalty',
    ]:
        args_kwargs[arg] = kwargs.pop(arg, getattr(args, arg))
    all_kwargs = {**kwargs, **args_kwargs}

    seed = all_kwargs.pop('seed', args.seed)

    env = DummyVecEnv([
        lambda: gym.make(
            args.envname,
            seed=seed + i if seed is not None else None,
            **all_kwargs,
        )
        for i in range(num_envs)
    ])
    if include_norm:
        # When training, set `norm_reward = True`, I hear...
        env = VecNormalize(
            env,
            norm_obs=norm_obs,
            norm_reward=norm_reward,
        )
    return env


def create_eval_callback(args, learning_rate):
    if args.eval_freq <= 0:
        return None

    seed = args.seed
    if seed is not None:
        seed += args.num_envs

    eval_env = make_env(
        args,
        num_envs=1,
        include_norm=True,
        # In contrast to training, we don't give this for testing.
        norm_reward=False,
        seed=seed,
    )

    best_dirname = Path(f'best_sdc_model_{args.model_class.lower()}_'
                        f'{args.policy_class.lower()}_{learning_rate}')
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dirname),
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
    )
    return eval_callback


def check_num_envs(args, policy_class):
    """Raise an error if the number of environments will cause issues with
    the model.
    """
    if not policy_class.recurrent:
        return
    if args.num_envs != 1:
        if args.eval_freq > 0:
            raise ValueError(
                'An `--eval_freq > 0` was given. Due to issues with recurrent '
                'policies and vectorized environments, the number of '
                'environments must be set to 1 (so append ` --num_envs 1`)'
            )

        print(
            'Warning: The model is recurrent and `--num_envs > 1` was given. '
            'Testing will take _much_ longer with this configuration than '
            'with `--num_envs 1`. Furthermore, due to an issue with recurrent '
            'policies and vectorized environments, the number of test '
            'environments must be the same as the number of '
            'training environments.'
        )


def maybe_fix_nminibatches(model_kwargs, args, policy_class):
    """Set the `nminibatches` parameter in `model_kwargs` to
    `args.num_envs` if the value would cause problems otherwise with the
    given `policy_class`.

    Print a warning as well if the value had to be changed.
    Only affects the `PPO2` model.
    """
    if args.model_class != 'PPO2' or not policy_class.recurrent:
        return

    nminibatches = model_kwargs.get(
        'nminibatches',
        PPO2_DEFAULT_NUM_MINIBATCHES,
    )
    if args.num_envs % nminibatches == 0:
        return

    print('Warning: policy is recurrent and the number of environments is '
          'not a multiple of `PPO2.nminibatches`. '
          'Setting `nminibatches = num_envs`...')
    model_kwargs['nminibatches'] = args.num_envs
