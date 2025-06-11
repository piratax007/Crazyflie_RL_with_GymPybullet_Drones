# Crazyflie_RL_with_GymPybullet_Drones
This repository contains environments, scripts, and utilities _developed apart from GymPybullet Drones [1]_ but used 
for running Reinforcement Learning training for Crazyflie drones using GymPybullet Drones.

## Structure
### assets
This directory is a copy of the assets in GymPybullet Drones intended to introduce modifications in the definition 
of the drones without altering the original installation.

### environments
In this directory should be added any new environment used for training, following the gymnasium [3] environments 
structure. This directory also contains a copy of the `BaseAviary` and `BaseRLAviary` environments, using these 
copies, you can modify in deep the basic training environments without alter the original ones.

### python_scripts
#### easy_plots
This is a module developed to produce nice publishable plots of the data generated during the evaluation of a trained 
policy in simulation. You can learn how to use easy plots in its README file [2].

#### classic_simulation.py
This module is used to simulate the Crazyflie drone using classic controllers as PID and Geometric Lee control. 
Under development!

#### learning_script.py
This module should be used through the `execute_sequential_learning.py` module, however, if you want to modify the 
structure of the neural network, the hyperparameters of the RL algorithm or add new callbacks, you should go for it 
into this module.

##### HPC
Examples of batch scripts could be found into the `Alvis_scripts` and `kebnekaise_scripts`.

#### execute_sequential_learning.py
This module is used to execute learning processes choosing between several RL algorithms (PPO, SAC, DDPG, TD3) 
available on SB3, callbacks, and training modes (from scratch, continuous). The following are examples of how to use 
this module.

##### How to
**Example 1:** Training from scratch with PPO for 30 million time steps (MTS)

```shell
python3 -m python_scripts.execute_sequential_learning --environment 'CLStage1Sim2Real' --learning-id 'learning_test' --algorithm 'ppo' --parallel-environments 4 --time-steps 30000000
```

> To avoid repetition, from now the examples will use the default arguments. You can see the default values for all 
the parameter with `python3 -m python_scripts.execute_sequential_leargnin -h`

**Example 2:** Training from scratch with PPO and stopping the training after the reward achieves a threshold.

```shell
python3 -m python_scripts.execute_sequential_learning --environment 'CLStage1Sim2Real' --learning-id 'training_with_reward_threshold' --stop-on-reward-threshold-flag True --reward-threshold 625
```

**Example 3:** Training from scratch with PPO and stopping the training after 150 episodes. Consider that each 
episode of training is different in length (number of time steps), at the beginning of the training the episodes are 
shorter.

```shell
python3 -m python_scripts.execute_sequential_learning --environment 'CLStage1Sim2Real' --learning-id 'training_for_150_episodes' --stop-on-max-episodes-flag True --stop-episodes 150
```

**Example 4:** Training in continuous stages initializing a new neural network with the weights and bias from a 
pre-trained neural network. This will transfer the adam stimator at the end of the previous training as well.

```shell
python3 -m python_scripts.execute_sequential_learning --environment 'CLStage2Sim2Real' --learning-id 'second_training_stage' --continuous-learning True --path-to-revious-model '/results/save-training_for_150_episodes-02.17.2025_09.18.11'
```

> If you add a new training environment, to use it for training, the new environment must be registered 
> in the `environments/__init__.py` file following the same patter as the currently available environments.

> The training is automatically monitored with tensorboard. You can find the tensor board. To see the tensorboard 
> logs run:
> `tensorboard --logdir path_to_the_results_folder/tb`

#### simulation_script.py
This module provides a command-line interface for running flight simulations of Crazyflie drones under RL trained 
policies. It automates the environment setup, policy loading, episode execution, and logging.

##### How to
**Example 1:** Running 10 seconds of simulation using a PPO trained policy
```shell
python3 -m python_scripts.simulation_script --polity_path results/some_trained_policy_directory --model 'best_model. zip' --algorithm 'ppo' --test_env 'some_registered_environment' --simulation-length 10
```
**Example 2:** Running 20 seconds of a policy checkpoint trained with SAC, saving logs of the observations and actions 
to CSV files, and generating some plots at the end of the simulation.
```shell
python3 -m python_scripts.simulation_script --policy_path results/policy_directory/checkpoints --model 'checkpoint_1000000_steps.zip' --algorithm 'sac' --test_env 'some_environment' --save True --comment 'logs_directory_name' --plot True
```
When the simulation is finished, is created a new directory `save_comment_date_hour` containing the CSV files for 
the logging data, and are generated odometry plots.

By default, the `simulation_script` uses PPO, and the simulation length is 20 seconds. The `--debug True` argument 
could be used to print to the console the observations and actions during the simulation and detailed information 
about the architecture of the neural network. You can use the `--reset True` argument, as well, if you want to reset 
the simulation when the drone achieves the termination conditions defined into the environment.

> The functions `helix_trajectory`, `lemniscate_trajectory`, and `smooth_trajectory` inside this script, could be 
> used to design evaluation trajectories for your trained policies. The function `random_cylindrical_positions` 
> could be used to assess the robustness of the trained policy to start from randon initial positions. These 
> functions, along with other useful functions in this script, are well documented in its docstrics.

##### HPC
If you want to run simulations on the HPC cluster, you should use a virtual desktop client (ThinLink or browser client).

The following example is designed to run on a login node on Kebnekaise
```bash
$ ml purge > /dev/null 2>&1
$ ml load GCC/12.3.0 OpenMPI/4.1.5
$ ml load gym-pybullet-drones/2.0.0-3d7b12edd4915a27e6cec9f2c0eb4b5479f7735e
$ ml load TensorFlow/2.15.1-CUDA-12.1.1
$ python3 -m python_scripts.simulation_script --polity_path results/some_trained_policy_directory --model 'best_model. zip' --algorithm 'ppo' --test_env 'some_registered_environment' --simulation-length 10
```
#### crazyflie_firmware_nn_compute
The modules inside this directory are used for export a policy trained using GymPybullet Drones to C code that can
be added to the Crazyflie firmware [4] to be deployed on real hardware.
- `c_code_blocks.py` contains the general definitions and the forward propagation loops. If the structure of the
  policy differs from 12 inputs - 64 nodes (tanh) - 64 nodes (tanh) - 4 outputs, you should edit this file in
  consequence.
- `crazyflie_firmware_nn_compute.py` This module is used for exporting to C code the weights and bias of a trained
  policy, and generate the `nn_compute.c` file that should replace the homonymous files in the Crazyflie firmware at
  `/examples/app_nn_controller/src`.
  - **How to:**
    - `python3 -m python_scripts.crazyflie_firmware_nn_compute.crazyflie_firmware_nn_compute 
    path_to_the_policy/best_model.zip -o path_to_the_directory_to_put/nn_compute.c -p true`
    - The `path_to_the_policy` is a mandatory argument pointing to the trained policy `best_model.zip`
    - The `-o` argument is optional pointing to the directory that will contain the `nn_compute.c` file. If it is not
      specified the `nn_compute.c` file will be saved in the current directory.
    - The `-p` optional argument will previsualice in console the content of the `nn_compute.c` file.

Once you have generated the `nn_compute.c` file, copy that file into the `examples/app_nn_controller/src` directory on
the Crazyflie firmware [4]. You are ready to test your trained policy on a real Crazyflie.

### crazyflie-firmware
This is a modified version of the Crazyflie firmware that implements an OOT controller which uses a neural network 
to map observations to PWM as control motor signals. More details about how it is implemented and how adapt it could 
be found in its documentation.

### crazyflie_demos
This is a ROS2 humble package with some nodes and scripts to control Crazyflie using different control strategies. 
More details about how it works could be found in its documentation.

---
[1]: https://github.com/utiasDSL/gym-pybullet-drones
[2]: python_scripts/easy_plots/README.md
[3]: https://gymnasium.farama.org/
[4]: https://github.com/piratax007/crazyflie-firmware