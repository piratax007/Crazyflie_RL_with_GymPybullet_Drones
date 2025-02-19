import matplotlib.pyplot as plt
from aquarel import load_theme
from normalize_reward_comparison import moving_average, read_from_csv, calculate_statistics

theme = (
        load_theme('scientific')
        .set_font(family='serif', size=22)
        .set_axes(bottom=True, top=True, left=True, right=True, xmargin=0, ymargin=0, zmargin=0, width=2)
        .set_grid(style='--', width=1)
        .set_ticks(draw_minor=True, pad_major=10)
        .set_lines(width=2.5)
        .set_legend(location='upper right', alpha=0)
    )
theme.apply()

ws = 15

base_path_CLS1 = 'CLS-1/Reported/'
CLS1_files = [
    base_path_CLS1 + 'save-CLS-1_ACC-EXTENSION_TRAIN-1_RT-7600-10.24.2024_21.34.49/run-tb_PPO_1-tag-eval_mean_reward.csv',
    base_path_CLS1 + 'save-CLS-1_ACC-EXTENSION_TRAIN-2_RT-7600-10.24.2024_21.33.57/run-tb_PPO_1-tag-eval_mean_reward.csv',
    base_path_CLS1 + 'save-CLS-1_ACC-EXTENSION_TRAIN-3_RT-7600-10.24.2024_21.32.56/run-tb_PPO_1-tag-eval_mean_reward.csv',
    base_path_CLS1 + 'save-CLS-1_ACC-EXTENSION_TRAIN-3_RT-7600-10.24.2024_21.32.56/run-tb_PPO_1-tag-eval_mean_reward.csv',
    base_path_CLS1 + 'save-CLS-1_WER_ACC-EXTENSION_TRAIN-5_RT-7600-10.24.2024_21.29.27/run-tb_PPO_1-tag-eval_mean_reward.csv',
]
CLS1_steps, CLS1_rewards = read_from_csv(CLS1_files)
CLS1_smoothed_mean_rewards = moving_average(calculate_statistics(CLS1_rewards)[0], ws)
CLS1_smoothed_std_rewards = moving_average(calculate_statistics(CLS1_rewards)[1], ws)
CLS1_smoothed_steps = CLS1_steps[ws - 1:]

base_path_CLS2 = 'CLS-2/reported/'
CLS2_files = [
    base_path_CLS2 + 'PPO_7/run-.-tag-eval_mean_reward.csv',
    base_path_CLS2 + 'PPO_8/run-.-tag-eval_mean_reward.csv',
    base_path_CLS2 + 'PPO_9/run-.-tag-eval_mean_reward.csv',
    base_path_CLS2 + 'PPO_10/run-.-tag-eval_mean_reward.csv',
    base_path_CLS2 + 'PPO_11/run-.-tag-eval_mean_reward.csv'
]
CLS2_steps, CLS2_rewards = read_from_csv(CLS2_files)
CLS2_smoothed_mean_rewards = moving_average(calculate_statistics(CLS2_rewards)[0], ws)
CLS2_smoothed_std_rewards = moving_average(calculate_statistics(CLS2_rewards)[1], ws)
CLS2_smoothed_steps = CLS2_steps[ws - 1:]

base_path_CLS3 = 'CLS_3/'
CLS3_files = [
    base_path_CLS3 + 'save-CLS-3_SIMPLE_TRAIN-1-10.28.2024_00.41.37/run-PPO_1-tag-eval_mean_reward.csv',
    base_path_CLS3 + 'save-CLS-3_SIMPLE_TRAIN-2-10.28.2024_00.42.43/run-PPO_2-tag-eval_mean_reward.csv',
    base_path_CLS3 + 'save-CLS-3_SIMPLE_TRAIN-3-10.28.2024_00.43.31/run-PPO_3-tag-eval_mean_reward.csv',
    base_path_CLS3 + 'save-CLS-3_SIMPLE_TRAIN-4-10.28.2024_00.44.13/run-PPO_4-tag-eval_mean_reward.csv',
    base_path_CLS3 + 'save-CLS-3_SIMPLE_TRAIN-5-10.28.2024_00.44.58/run-PPO_5-tag-eval_mean_reward.csv'
]
CLS3_steps, CLS3_rewards = read_from_csv(CLS3_files)
CLS3_smoothed_mean_rewards = moving_average(calculate_statistics(CLS3_rewards)[0], ws)
CLS3_smoothed_std_rewards = moving_average(calculate_statistics(CLS3_rewards)[1], ws)
CLS3_smoothed_steps = CLS3_steps[ws - 1:]

# base_path_WCL = 'Without-CL/Reported/'
# WCL_files = [
#     base_path_WCL + 'save-WCL_ACC-EXTENSION_TRAIN-1-10.26.2024_00.03.38/run-tb_PPO_1-tag-eval_mean_reward.csv',
#     base_path_WCL + 'save-WCL_ACC-EXTENSION_TRAIN-2-10.26.2024_00.04.25/run-tb_PPO_1-tag-eval_mean_reward.csv',
#     base_path_WCL + 'save-WCL_ACC-EXTENSION_TRAIN-3-10.26.2024_00.04.54/run-tb_PPO_1-tag-eval_mean_reward.csv',
#     base_path_WCL + 'save-WCL_ACC-EXTENSION_TRAIN-4-10.26.2024_00.05.43/run-tb_PPO_1-tag-eval_mean_reward.csv',
#     base_path_WCL + 'save-WCL_ACC-EXTENSION_TRAIN-5-10.26.2024_00.06.48/run-tb_PPO_1-tag-eval_mean_reward.csv'
# ]
base_path_WCL = 'Without-CL/'
WCL_files = [
    base_path_WCL + 'save-WITHOUT_CL-SEED_NONE-TRAIN_1-10.20.2024_00.25.40/run-tb_PPO_1-tag-eval_mean_reward.csv',
    base_path_WCL + 'save-WITHOUT_CL-SEED_NONE-TRAIN_2-10.20.2024_00.25.39/run-tb_PPO_1-tag-eval_mean_reward.csv',
    base_path_WCL + 'save-WITHOUT_CL-SEED_NONE-TRAIN_3-10.20.2024_00.25.39/run-tb_PPO_1-tag-eval_mean_reward.csv',
    base_path_WCL + 'save-WITHOUT_CL-SEED_NONE-TRAIN_4-10.20.2024_00.25.57/run-tb_PPO_1-tag-eval_mean_reward.csv',
    base_path_WCL + 'save-WITHOUT_CL-SEED_NONE-TRAIN_4-10.20.2024_00.26.06/run-tb_PPO_1-tag-eval_mean_reward.csv'
]
WCL_steps, WCL_rewards = read_from_csv(WCL_files)
WCL_smoothed_mean_rewards = moving_average(calculate_statistics(WCL_rewards)[0], ws)
WCL_smoothed_std_rewards = moving_average(calculate_statistics(WCL_rewards)[1], ws)
WCL_smoothed_steps = WCL_steps[ws - 1:]

plt.rcParams['text.usetex'] = True
# plt.plot(CLS1_smoothed_steps, CLS1_smoothed_mean_rewards, color='forestgreen', label='$\Omega 1$')
# plt.plot(CLS2_smoothed_steps, CLS2_smoothed_mean_rewards, color='cornflowerblue', label='$\Omega 2$')
# plt.plot(CLS3_smoothed_steps, CLS3_smoothed_mean_rewards, color='lightsalmon', label='$\Omega 3$')
plt.plot(WCL_smoothed_steps, WCL_smoothed_mean_rewards, color='red', label='Without CL')
# plt.fill_between(CLS1_smoothed_steps, CLS1_smoothed_mean_rewards - CLS1_smoothed_std_rewards, CLS1_smoothed_mean_rewards + CLS1_smoothed_std_rewards, color='forestgreen', alpha=0.2)
# plt.fill_between(CLS2_smoothed_steps, CLS2_smoothed_mean_rewards - CLS2_smoothed_std_rewards, CLS2_smoothed_mean_rewards + CLS2_smoothed_std_rewards, color='cornflowerblue', alpha=0.2)
# plt.fill_between(CLS3_smoothed_steps, CLS3_smoothed_mean_rewards - CLS3_smoothed_std_rewards, CLS3_smoothed_mean_rewards + CLS3_smoothed_std_rewards, color='lightsalmon', alpha=0.2)
plt.fill_between(WCL_smoothed_steps, WCL_smoothed_mean_rewards - WCL_smoothed_std_rewards, WCL_smoothed_mean_rewards + WCL_smoothed_std_rewards, color='red', alpha=0.2)
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.xlabel('Time Steps', labelpad=15)
plt.xlim(right=30_000_000)
plt.ylabel('ECR', labelpad=15)
plt.legend(bbox_to_anchor=(0, 1, 1, 0.25), loc="lower right", borderaxespad=0.5, ncol=4)
plt.show()