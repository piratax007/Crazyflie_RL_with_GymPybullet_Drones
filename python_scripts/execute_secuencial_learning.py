#!/usr/bin/env python3
from environments.ObS12Stage1 import ObS12Stage1
from python_scripts.learning_script import run_learning

print("""############# Base training #############
Starting = [0 0 0 0 0 0]
Target = [0 0 1 Nan Nan Nan]
##################################################
""")

results = run_learning(environment=ObS12Stage1,
                       learning_id="ACTION-SPACE-VEL",
                       continuous_learning=False,
                       parallel_environments=4,
                       time_steps=int(10e7),
                       stop_on_max_episodes=dict(stop=False, episodes=0),
                       stop_on_reward_threshold=dict(stop=True, threshold=610.),
                       save_checkpoints=dict(save=True, save_frequency=250000)
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")
