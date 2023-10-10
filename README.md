# PTSA-MCTS
A PyTorch implementation of PTSA-MCTS from [Accelerating Monte Carlo Tree Search with
Probability Tree State Abstraction].

*  Is modified based on [model-based-rl](https://github.com/JimOhman/model-based-rl)

## Reproduce examples:

* Pong-ramNoFrameskip-v4: ```python train.py --environment Pong-ramNoFrameskip-v4 --architecture FCNetwork --num_actors 7 
--fixed_temperatures 1.0 0.8 0.7 0.5 0.3 0.2 0.1 --td_steps 10 --obs_range 0 255 --norm_obs --sticky_actions 4 --noop_reset --episode_life  --group_tag my_group_tag --run_tag my_run_tag```


