#!/usr/bin/env python3
from environments.ObS12Stage3 import ObS12Stage3
from python_scripts.learning_script import run_learning

print("""############# Base training #############
Starting = [0 0 0 0 0 0]
Target = [0 0 1 Nan Nan Nan]
##################################################
""")

results = run_learning(environment=ObS12Stage3,
                       learning_id="WCL-RANDOM_VELOCITIES-ACC",
                       continuous_learning=False,
                       parallel_environments=50,
                       time_steps=int(30e6),
                       stop_on_max_episodes=dict(stop=False, episodes=0),
                       stop_on_reward_threshold=dict(stop=False, threshold=600.),
                       save_checkpoints=dict(save=True, save_frequency=20000)
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")
