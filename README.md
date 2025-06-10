# Crazyflie_RL_with_GymPybullet_Drones
This repository contains environments, scripts, and utilities _developed apart from GymPybullet Drones_ but used 
for running Reinforcement Learning training for Crazyflie drones using GymPybullet Drones.

## Structure
### assets
This directory is a copy of the assets in GymPybullet Drones intended to introduce modifications in the definition 
of the drones without altering the original installation.

### environments
In this directory should be added any new environment used for training, following the gymnasium environments 
structure. This directory also contains a copy of the `BaseAviary` and `BaseRLAviary` environment, to be able of 
adding training features without altering the original installation of GymPybullet Drones.

### python_scripts
#### crazyflie_firmware_nn_compute
The modules inside this directory are used for export a policy trained using GymPybullet Drones to C code that can 
be added to the Crazyflie firmware to be deployed on real hardware.
- `c_code_blocks.py` contains the general definitions and the forward propagation loops. If the structure of the 
  policy differs from 12 inputs - 64 nodes (tanh) - 64 nodes (tanh) - 4 outputs, you should edit this file in 
  consequence.
- `crazyflie_firmware_nn_compute.py` This module is used for exporting to C code the weights and bias of a trained 
  policy, and generate the `nn_compute.c` file that should be added to the Crazyflie firmware. 
  - **How to:**
    - `python3 -m python_scripts.crazyflie_firmware_nn_compute.crazyflie_firmware_nn_compute 
    path_to_the_policy/best_model.zip -o path_to_the_directory_to_put/nn_compute.c -p true`
    - The `path_to_the_policy` is a mandatory argument pointing to the trained policy `best_model.zip`
    - The `-o` argument is optional pointing to the directory that will contain the `nn_compute.c` file. If it is not 
      specified the `nn_compute.c` file will be saved in the current directory.
    - The `-p` optional argument will previsualice in console the content of the `nn_compute.c` file.

Once you have generated the `nn_compute.c` file, copy that file into the `examples/app_nn_controller` directory on 
the Crazyflie firmware. You are ready to test your trained policy on a real Crazyflie.

### easy_plots
This is a module developed to produce nice publishable plots of the data generated during the evaluation of a trained 
policy in simulation. You can learn how to use easy plots in its README file [2].

### classic_simulation.py
This module is used to simulate the Crazyflie drone using classic controllers as PID and Geometric Lee control. 
Under development!

### learning_script.py
This module should be used through the `execute_sequential_learning.py` module, however, if you want to modify the 
structure of the neural network, the hyperparameters of the RL algorithm or add new callbacks, you should go for it 
into this module.

### execute_sequential_learning.py
This module is used to execute learning processes choosing between several RL algorithms (PPO, SAC, DDPG, TD3) 
available on SB3, callbacks, and training modes (from scratch, continuous). The following are examples of how to use 
this module.

**Example 1:** Training from scratch with PPO for 30 million time steps (MTS)

```python3 -m python_scripts.execute_sequential_learning --environment 'CLStage1Sim2Real' --learning-id 'learning_test' --algorithm 'ppo' --parallel-environments 4 --time-steps 30000000```

> To avoid repetition, from now the examples will use the default parameters. You can see the default values for all 
the parameter with `python3 -m python_scripts.execute_sequential_leargnin -h`

**Example 2:** Training from scratch with PPO and stopping the training after the reward achieves a threshold.

```python3 -m python_scripts.execute_sequential_learning --environment 'CLStage1Sim2Real' --learning-id 'training_with_reward_threshold' --stop-on-reward-threshold-flag True --reward-threshold 625```

**Example 3:** Training from scratch with PPO and stopping the training after 150 episodes. Consider that each 
episode of training is different in length (number of time steps), at the beginning of the training the episodes are 
shorter.

```python3 -m python_scripts.execute_sequential_learning --environment 'CLStage1Sim2Real' --learning-id 'training_for_150_episodes' --stop-on-max-episodes-flag True --stop-episodes 150```

**Example 4:** Training in continuous stages initializing a new neural network with the weights and bias from a 
pre-trained neural network. This will transfer the adam stimator at the end of the previous training as well.

```python3 -m python_scripts.execute_sequential_learning --environment 'CLStage2Sim2Real' --learning-id 'second_training_stage' --continuous-learning True --path-to-revious-model '/results/save-training_for_150_episodes-02.17.2025_09.18.11'```

> If you add a new training environment, to be able to use it for training, the new environment must be registered 
> in the `execute_sequential_learning` module. To register a new environment, import it, add the name of the class 
> into the `choices` list of the `--environment` argument, and add a new entry in the `environment_map` dictionary.

### simulation_script.py
This module provides a command-line interface for running flight simulations of Crazyflie drones under RL trained 
policies. It automates the environment setup, policy loading, episode execution, and logging.

**How to**
...

---
[1]: https://github.com/utiasDSL/gym-pybullet-drones
[2]: python_scripts/easy_plots/README.md