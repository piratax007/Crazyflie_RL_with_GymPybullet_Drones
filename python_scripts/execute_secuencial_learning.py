#!/usr/bin/env python3
from gym_pybullet_drones.envs import HoverCrazyflieSim2Real
from gym_pybullet_drones.experiments.learning_script import run_learning

print("""############# Base training #############
Starting = [0 0 0 0 0 0]
Target = [0 0 1 Nan Nan Nan]
##################################################
""")

results = run_learning(environment=HoverCrazyflieSim2Real,
                       learning_id="HOVER_NOISED_TRAINING-02",
                       continuous_learning=False,
                       parallel_environments=4,
                       time_steps=int(100e6),
                       stop_on_max_episodes=dict(stop=False, episodes=0),
                       stop_on_reward_threshold=dict(stop=True, threshold=600.),
                       save_checkpoints=dict(save=True, save_frequency=25000)
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")
