# sdc-gym

## Differentiable Programming RL

The main script is [./dp_playground.py](./dp_playground.py) or its
variants. For command line arguments, see the according `parse_args`
function.

Modify the `build_model` function according to the desired
architecture. Adjusting the learning rate schedule in the `build_opt`
function may also prove worthwhile.

### Training

#### Standard Training

While the best models are obtained after multiple learning rate waves,
training may also be stopped after 30000 steps. An example training
command looks like this:

```shell
python dp_playground.py --M 5 --steps 200000 --batch_size 32 \
       --lambda_real_interval -100 0 --lambda_imag_interval -10 0
```

#### Continuing Training

Training can be continued using the `--model_path` argument. Simply
load a model checkpoint and continue as desired. Note that the
optimizer state is _not_ included in saved checkpoints, so it is
recommended to schedule the learning rate so training only starts
after the optimizer has adapted a bit (when using an adaptive one).
Example usage:

```shell
python dp_playground.py --M 5 --steps 200000 --batch_size 32 \
       --lambda_real_interval -100 0 --lambda_imag_interval -10 0 \
       --model_path best_dp_model_diag_M_5_re_-100.0_0.0_im_-10.0_0.0_loss_[...].npy
```

#### Optimizing Parameters Directly

To obtain a model that optimizes the preconditioner's parameters
directly, you would give the argument `--optimize_directly True`.

An example training to optimize for a single λ value with M = 5 would
be started like this. Note also that we set the batch size to 1 to
avoid redundant work:

```shell
python dp_playground.py --M 5 --steps 200000 --optimize_directly True \
       --batch_size 1 --lambda_real_interval -1 -1 --lambda_imag_interval 0 0
```

To optimize a strictly lower triangular preconditioner, testing on
some additional preconditioners:

```shell
python dp_playground.py --steps 200000 --optimize_directly True \
       --batch_size 1 --prec_type strictly_lower_tri --extensive_tests True \
       --lambda_real_interval -1 -1 --lambda_imag_interval 0 0
```

### Evaluation

#### Using Training Script for Evaluation Only

Simply set the number of training steps to 0 (`--steps 0`) – the
(possibly loaded) model will be used as is.

### Sharing Models

The files to share when sharing a model for continued training are
those ending in `.npy`, `.structure` and `.steps`. When only
interested in interference, the file ending in `.steps` does not need
to be shared.

## Reinforcement Learning

The main script is [./rl_playground.py](./rl_playground.py). For
command line arguments, see
[./utils/arguments.py](./utils/arguments.py). There are some
recommended defaults to set below.

The command line arguments given are automatically saved upon script
start. Most files saved are stored with the starting time of the
script as a timestamp, so all files for one experiment should be
immediately recognizable by having the same timestamp (except for
TensorBoard logs for now).

### Recommended arguments

```shell
python rl_playground.py --envname sdc-v1 --num_envs 8 \
       --model_class PPG --activation_fn ReLU \
       --collect_states True --reward_iteration_only False --norm_obs True
```

### Another recommendation

To accelerate learning, increase the batch size if possible. Here is
an example for PPG:

PPG has a default batch size of 64 (given as the keyword argument
`batch_size`), so we could use a batch size of 512 like this:

```shell
python rl_playground.py --model_class PPG \
       --model_kwargs '{"batch_size": 512}'
```

This will, however, most likely harm your training success due to
executing fewer training steps, as we process much more data (and thus
more environmental timesteps) in each training step. A good heuristic
to help against this problem is scaling the learning rate
proportionally to the batch size. The default learning rate we give is
25e-5, so scaling it to the increased batch size is `25e-5 * 512 / 64
= 0.002`. Our new command for starting the script becomes the
following:

```shell
python rl_playground.py --model_class PPG --learning_rate 0.002 \
       --model_kwargs '{"batch_size": 512}'
```

For PPG, keep in mind that it also uses an auxiliary batch size
(`aux_batch_size`)! Half of the normal batch size is a good starting
value for this. The final command is:

```shell
python rl_playground.py --model_class PPG --learning_rate 0.002 \
       --model_kwargs '{"batch_size": 512, "aux_batch_size": 256}'
```
