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
