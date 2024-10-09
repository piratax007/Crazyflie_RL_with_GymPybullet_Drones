#!/usr/bin/env python3
from environments.CurriculumStage1 import CurriculumStage1
from python_scripts.learning_script import run_learning

print("""############# Base training #############
Starting = [0 0 0 0 0 0]
Target = [0 0 1 Nan Nan Nan]
##################################################
""")

results = run_learning(environment=CurriculumStage1,
                       learning_id="Curriculum-Stage-1_Without-noise_rew-800",
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
