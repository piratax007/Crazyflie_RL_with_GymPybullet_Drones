#!/usr/bin/env python3
from environments.HoverCrazyflieSim2Real import HoverCrazyflieSim2Real
from python_scripts.learning_script import run_learning

print("""############# Base training #############
Starting = [0 0 0 0 0 0]
Target = [0 0 1 Nan Nan Nan]
##################################################
""")

results = run_learning(environment=HoverCrazyflieSim2Real,
                       learning_id="Sim2Real_CL_SUB-TASK_1_rew-800_env-4_sed-42_no-cuda",
                       continuous_learning=False,
                       parallel_environments=4,
                       time_steps=int(100e6),
                       stop_on_max_episodes=dict(stop=False, episodes=0),
                       stop_on_reward_threshold=dict(stop=True, threshold=800.),
                       save_checkpoints=dict(save=True, save_frequency=250000)
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")
