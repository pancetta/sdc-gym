import json
from pathlib import Path

import gym

import sdc_gym  # For side effects only
import ppg

PPO2_DEFAULT_NUM_MINIBATCHES = 4


def setup(using_sb3, debugging_nans=False):
    global use_sb3, debug_nans, \
        CheckpointCallback, EvalCallback, \
        DummyVecEnv, VecCheckNan, VecNormalize, \
        stable_baselines, act_fns

    use_sb3 = using_sb3
    debug_nans = debugging_nans

    if use_sb3:
        from stable_baselines3.common.callbacks import CheckpointCallback, \
            EvalCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, \
            VecCheckNan, VecNormalize
        import stable_baselines3 as stable_baselines
        import torch.nn as act_fns
        if debug_nans:
            import torch
            torch.autograd.set_detect_anomaly(True)
    else:
        from stable_baselines.common.callbacks import CheckpointCallback, \
            EvalCallback
        from stable_baselines.common.vec_env import DummyVecEnv, \
            VecCheckNan, VecNormalize
        from stable_baselines.common import tf_layers as act_fns
        import stable_baselines


def has_sb3():
    try:
        import stable_baselines3
    except ModuleNotFoundError:
        return False
    return True


def parse_bool(string):
    assert string == 'False' or string == 'True', \
        'please only use "False" or "True" as boolean arguments.'
    return string != 'False'


def _json_handle_constant(python_str, json_str):

    def handle_constant(string, pos):
        if (
                string[pos] == python_str[0]
                and string[pos:pos + len(python_str)] == python_str
        ):
            return string[:pos] + json_str + string[pos + len(python_str):]
        return string

    return handle_constant


def _json_fix_string(string, ex):
    json_err_char_to_handler = {
        'T': _json_handle_constant('True', 'true'),
        'F': _json_handle_constant('False', 'false'),
        'N': _json_handle_constant('None', 'null'),
    }

    err_pos = ex.pos
    err_char = string[err_pos]
    if err_char not in json_err_char_to_handler:
        raise ex

    shape_string = json_err_char_to_handler[err_char]
    string = shape_string(string, err_pos)


def parse_dict(string):
    while True:
        try:
            return json.loads(string)
        except json.JSONDecodeError as ex:
            string = _json_fix_string(string, ex)


def get_model_class(model_class_str):
    """Return a model class according to `model_class_str`."""
    if model_class_str.upper() == 'PPG':
        if not use_sb3:
            raise AttributeError(
                'PPG is only implemented in stable-baselines3. '
                'Please set `use_sb3 = True`.'
            )
        return ppg.PPG

    if use_sb3 and model_class_str.upper() == 'PPO2':
        model_class_str = 'PPO'

    try:
        model_class = getattr(stable_baselines, model_class_str)
    except AttributeError:
        raise AttributeError(
            f"could not find model class '{model_class_str}' "
            f'in module `stable_baselines{"3" if use_sb3 else ""}`'
        )
    if use_sb3:
        assert issubclass(
            model_class, stable_baselines.common.base_class.BaseAlgorithm), \
            ('model class must be a subclass of '
             '`stable_baselines3.common.base_class.BaseAlgorithm`')
    else:
        assert issubclass(model_class, stable_baselines.common.BaseRLModel), \
            ('model class must be a subclass of '
             '`stable_baselines.common.BaseRLModel`')
    return model_class


def get_policy_class(policy_class_str, model_class_str):
    """Return a policy class according to `policy_class_str`.
    `model_class_str` is the model to create the policy for.
    """
    if use_sb3:
        if model_class_str.upper() == 'PPG':
            return 'Aux' + policy_class_str
        return policy_class_str

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


def get_activation_fn(activation_fn_str):
    "Return an activation function given by `activation_fn_str`."
    try:
        activation_fn = getattr(act_fns, activation_fn_str)
    except AttributeError:
        err_string = (f'could not find activation function '
                      f"'{activation_fn_str}' in module ")
        if use_sb3:
            err_string = f'{err_string}`torch.nn`'
        else:
            err_string = f'{err_string}`stable_baselines.common.tf_layers`'
        raise AttributeError(err_string)
    return activation_fn


def compute_learning_rate(args):
    learning_rate = args.learning_rate
    if args.rescale_lr:
        learning_rate *= args.num_envs
    return learning_rate


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
        norm_reward=True,
        **kwargs,
):
    """Return a vectorized environment containing `num_envs` or `args.num_envs`
    environments (depending on whether `num_envs is None`).

    `args`, the command line arguments, specify several values. See `kwargs`
    for a more detailed explanation on their interaction.
    `include_norm` specifies whether the environment is wrapped in a
    normalizing environment.
    `norm_reward` indicates whether the rewards are normalized (only
    revelant if `include_norm is True`).
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

            'lambda_real_interval',
            'lambda_imag_interval',
            'lambda_real_interpolation_interval',

            'norm_factor',
            'residual_weight',
            'step_penalty',
            'reward_iteration_only',
            'collect_states',
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
        if args.env_path is not None:
            env = VecNormalize(str(Path(args.env_path)), env)
        else:
            # When training, set `norm_reward = True`, I hear...
            if 'gamma' in args.model_kwargs:
                env = VecNormalize(
                    env,
                    norm_obs=args.norm_obs,
                    norm_reward=norm_reward,
                    gamma=args.model_kwargs['gamma'],
                )
            else:
                env = VecNormalize(
                    env,
                    norm_obs=args.norm_obs,
                    norm_reward=norm_reward,
                )
    if debug_nans:
        env = VecCheckNan(env, raise_exception=True)
    return env


def create_save_callback(args):
    if args.save_freq <= 0:
        return None

    dirname = Path(f'sdc_model_{args.model_class.lower()}_'
                   f'{args.policy_class.lower()}_{args.script_start}')
    callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(dirname),
    )
    return callback


def create_eval_callback(args):
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
                        f'{args.policy_class.lower()}_{args.script_start}')
    callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dirname),
        eval_freq=args.eval_freq,
        n_eval_episodes=50,
        deterministic=True,
        render=False,
    )
    return callback


def append_callback(callbacks, callback):
    if callback is not None:
        callbacks.append(callback)
    return callbacks


def check_num_envs(args, policy_class):
    """Raise an error if the number of environments will cause issues with
    the model.
    """
    if use_sb3 or not policy_class.recurrent:
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
    if args.model_class != 'PPO2' or use_sb3 or not policy_class.recurrent:
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


def save_env(path, env):
    if isinstance(env, VecNormalize):
        env.save(str(path))
