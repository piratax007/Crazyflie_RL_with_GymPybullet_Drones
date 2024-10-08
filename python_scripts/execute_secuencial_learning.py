#!/usr/bin/env python3
from environments.WithoutCurriculumLearning import WithoutCurriculumLearning
from python_scripts.learning_script import run_learning

print("""############# Base training #############
Starting = [0 0 0 0 0 0]
Target = [0 0 1 Nan Nan Nan]
##################################################
""")

results = run_learning(environment=WithoutCurriculumLearning,
                       learning_id="Training_Without-Curriculum-Learning_standard-hyperparameters_sed-42",
                       continuous_learning=False,
                       parallel_environments=4,
                       time_steps=int(100e6),
                       stop_on_max_episodes=dict(stop=False, episodes=0),
                       stop_on_reward_threshold=dict(stop=False, threshold=800.),
                       save_checkpoints=dict(save=True, save_frequency=250000)
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")
