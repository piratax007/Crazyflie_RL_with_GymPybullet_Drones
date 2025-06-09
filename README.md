# Crazyflie_RL_with_GymPybullet_Drones
This repository contains environments, scripts, and utilities _developed apart from GymPybullet Drones_ but used 
for running Reinforcement Learning training for Crazyflie drones using GymPybullet Drones.

## Structure
### HPC scripts
**`Alvis_scripts`** and **`kebnekaise_scripts`** directories contains the run the training on HPC cluster. How 
those scripts work will be explained later.

> The `Python_scripts` still working on a local installation of GymPybullet Drones [1].

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
This is a module developed for produce nice published plots of the data generated during the evaluation of a trained 
policy in simulation. You can learn how to use easy plots in its README file [2].

### classic_simulation.py
This module is used to simulate the Crazyflie drone using classic controllers as PID and Geometric Lee control. 
Under development!

### learning_script.py
This module should be used through the `execute_sequential_learning.py` module, however, if you want to modify the 
structure of the neural network, the hyperparameters of the RL algorithm or add new callbacks, you should go for it 
into this module.

### simulation_script.py
This module provides a command-line interface for running flight simulations of Crazyflie drones under RL trained 
policies. It automates the environment setup, policy loading, episode execution, and logging.

**How to**
...

---
[1]: https://github.com/utiasDSL/gym-pybullet-drones
[2]: python_scripts/easy_plots/README.md