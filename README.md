# sdc-gym

The main script is [./rl_playground.py](./rl_playground.py). For
command line arguments, see
[./utils/arguments.py](./utils/arguments.py). There are some
recommended defaults to set below.

The command line arguments given are automatically saved upon script
start. Most files saved are stored with the starting time of the
script as a timestamp, so all files for one experiment should be
immediately recognizable by having the same timestamp (except for
TensorBoard logs for now).

## Recommended arguments

```shell
python rl_playground.py --envname sdc-v1 --num_envs 8 \
       --model_class PPG --activation_fn ReLU \
       --collect_states True --reward_iteration_only False --norm_obs True
```

## Another recommendation

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
25e-5, so scaling it to the increased batch size is `25e5 * 512 / 64 =
0.002`. Our new command for starting the script becomes the following:

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
