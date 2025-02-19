import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from aquarel import load_theme

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

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size


def read_from_csv(files: list):
    all_values = []
    steps = None

    for file in files:
        data = pd.read_csv(file)
        if steps is None:
            steps = data['Step']
        all_values.append(data['Value'])

    return steps, pd.DataFrame(all_values)

def calculate_statistics(rewards: pd.DataFrame):
    mean_rewards = rewards.mean(axis=0)
    std_rewards = rewards.std(axis=0)

    return mean_rewards, std_rewards

ws = 100

################ FINE TUNING ##################
NRN_csv_files = [
    'fine_tuning/No_normalized_reward/save-CLS_1-NO_REWARD_NORMALIZATION-SEED_0-FT-HF-10.17.2024_22.41.49/run-PPO_1-tag-rollout_ep_rew_mean.csv',
    'fine_tuning/No_normalized_reward/save-CLS_1-NO_REWARD_NORMALIZATION-SEED_39-FT-HF-10.17.2024_23.45.12/run-PPO_1-tag-rollout_ep_rew_mean.csv',
    'fine_tuning/No_normalized_reward/save-CLS_1-NO_REWARD_NORMALIZATION-SEED_39-FT-HF-10.17.2024_23.45.46/run-PPO_1-tag-rollout_ep_rew_mean.csv',
    'fine_tuning/No_normalized_reward/save-CLS_1-NO_REWARD_NORMALIZATION-SEED_42-FT-HF-10.17.2024_23.44.15/run-PPO_1-tag-rollout_ep_rew_mean.csv',
    'fine_tuning/No_normalized_reward/save-CLS_1-NO_REWARD_NORMALIZATION-SEED_90-FT-HF-10.17.2024_23.44.40/run-PPO_1-tag-rollout_ep_rew_mean.csv',
]

NRN_steps, NRN_rewards = read_from_csv(NRN_csv_files)
NRN_rewards_mean, NRN_rewards_std = calculate_statistics(NRN_rewards)
NRN_smoothed_mean_rewards = moving_average(NRN_rewards_mean, ws)
NRN_smoothed_std_rewards = moving_average(NRN_rewards_std, ws)
NRN_smoothed_steps = NRN_steps[ws - 1:]

NR_csv_files = [
    'fine_tuning/Normalized_reward/save-CLS_1-NORMALIZED_REWARD-SEED_0-FT-HF-10.17.2024_22.39.10/run-PPO_1-tag-rollout_ep_rew_mean.csv',
    'fine_tuning/Normalized_reward/save-CLS_1-NORMALIZED_REWARD-SEED_39-FT-HF-10.17.2024_23.41.32/run-PPO_1-tag-rollout_ep_rew_mean.csv',
    'fine_tuning/Normalized_reward/save-CLS_1-NORMALIZED_REWARD-SEED_42-FT-HF-10.17.2024_23.41.06/run-tb_PPO_1-tag-rollout_ep_rew_mean.csv',
    'fine_tuning/Normalized_reward/save-CLS_1-NORMALIZED_REWARD-SEED_90-FT-HF-10.17.2024_23.41.06/run-tb_PPO_1-tag-rollout_ep_rew_mean.csv',
    'fine_tuning/Normalized_reward/save-CLS_1-NORMALIZED_REWARD-SEED_1234-FT-HF-10.17.2024_23.40.19/run-tb_PPO_1-tag-rollout_ep_rew_mean.csv'
]

NR_steps, NR_rewards = read_from_csv(NR_csv_files)
NR_rewards_mean, NR_rewards_std = calculate_statistics(NR_rewards)
NR_smoothed_mean_rewards = moving_average(NR_rewards_mean, ws)
NR_smoothed_std_rewards = moving_average(NR_rewards_std, ws)
NR_smoothed_steps = NR_steps[ws - 1:]

############## WITHOUT FINE TUNING ######################
WFT_NRN_files = [
    'Without_fine_tuning/No_normalized_reward/save-CLS_1-NO_NORMALIZED_REWARD-SH-SEED_0-HF-10.17.2024_22.58.25/run-tb_PPO_1-tag-rollout_ep_rew_mean.csv',
    'Without_fine_tuning/No_normalized_reward/save-CLS_1-NO_NORMALIZED_REWARD-SH-SEED_39-HF-10.17.2024_23.27.11/run-tb_PPO_1-tag-rollout_ep_rew_mean.csv',
    'Without_fine_tuning/No_normalized_reward/save-CLS_1-NO_NORMALIZED_REWARD-SH-SEED_42-HF-10.17.2024_23.03.01/run-tb_PPO_1-tag-rollout_ep_rew_mean.csv',
    'Without_fine_tuning/No_normalized_reward/save-CLS_1-NO_NORMALIZED_REWARD-SH-SEED_90-HF-10.17.2024_23.26.44/run-tb_PPO_1-tag-rollout_ep_rew_mean.csv',
    'Without_fine_tuning/No_normalized_reward/save-CLS_1-NO_NORMALIZED_REWARD-SH-SEED_1234-HF-10.17.2024_23.00.45/run-tb_PPO_1-tag-rollout_ep_rew_mean.csv'
]

WFT_NRN_steps, WFT_NRN_rewards = read_from_csv(WFT_NRN_files)
WFT_NRN_rewards_mean, WFT_NRN_rewards_std = calculate_statistics(WFT_NRN_rewards)
WFT_NRN_smoothed_mean_rewards = moving_average(WFT_NRN_rewards_mean, ws)
WFT_NRN_smoothed_std_rewards = moving_average(WFT_NRN_rewards_std, ws)
WFT_NRN_smoothed_steps = WFT_NRN_steps[ws - 1:]

WFT_NR_files = [
    'Without_fine_tuning/Normalized_reward/save-CLS_1-NORMALIZED_REWARD-SH-SEED_0-HF-10.17.2024_23.28.58/run-tb_PPO_1-tag-rollout_ep_rew_mean.csv',
    'Without_fine_tuning/Normalized_reward/save-CLS_1-NORMALIZED_REWARD-SH-SEED_39-HF-10.17.2024_23.30.20/run-tb_PPO_1-tag-rollout_ep_rew_mean.csv',
    'Without_fine_tuning/Normalized_reward/save-CLS_1-NORMALIZED_REWARD-SH-SEED_42-HF-10.17.2024_23.29.33/run-tb_PPO_1-tag-rollout_ep_rew_mean.csv',
    'Without_fine_tuning/Normalized_reward/save-CLS_1-NORMALIZED_REWARD-SH-SEED_90-HF-10.17.2024_23.30.20/run-tb_PPO_1-tag-rollout_ep_rew_mean.csv',
    'Without_fine_tuning/Normalized_reward/save-CLS_1-NORMALIZED_REWARD-SH-SEED_1234-HF-10.17.2024_23.28.58/run-tb_PPO_1-tag-rollout_ep_rew_mean.csv'
]

WFT_NR_steps, WFT_NR_rewards = read_from_csv(WFT_NR_files)
WFT_NR_rewards_mean, WFT_NR_rewards_std = calculate_statistics(WFT_NR_rewards)
WFT_NR_smoothed_mean_rewards = moving_average(WFT_NR_rewards_mean, ws)
WFT_NR_smoothed_std_rewards = moving_average(WFT_NR_rewards_std, ws)
WFT_NR_smoothed_steps = WFT_NR_steps[ws - 1:]

if __name__ == '__main__':
    plt.rcParams['text.usetex'] = True
    plt.plot(NRN_smoothed_steps, NRN_smoothed_mean_rewards, color='forestgreen' ,label='Hyperparameter fine-tuning')
    plt.fill_between(NRN_smoothed_steps, NRN_smoothed_mean_rewards - NRN_smoothed_std_rewards, NRN_smoothed_mean_rewards + NRN_smoothed_std_rewards, color='forestgreen', alpha=0.2)
    plt.plot(WFT_NRN_smoothed_steps, WFT_NRN_smoothed_mean_rewards, color='steelblue', label='Default hyperparameters')
    plt.fill_between(WFT_NRN_smoothed_steps, WFT_NRN_smoothed_mean_rewards - WFT_NRN_smoothed_std_rewards, WFT_NRN_smoothed_mean_rewards + WFT_NRN_smoothed_std_rewards, color='steelblue', alpha=0.2)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.xlabel('Time Steps', labelpad=15)
    plt.ylabel('Episode Mean Reward', labelpad=15)
    plt.legend(bbox_to_anchor=(0, 1, 1, 0.25), loc="lower right", borderaxespad=0.5, ncol=4)
    plt.show()

    plt.plot(NR_smoothed_steps, NR_smoothed_mean_rewards, label='Hyperparameter fine-tuning', color='forestgreen')
    plt.fill_between(NR_smoothed_steps, NR_smoothed_mean_rewards - NR_smoothed_std_rewards, NR_smoothed_mean_rewards + NR_smoothed_std_rewards, color='forestgreen', alpha=0.2)
    plt.plot(WFT_NR_smoothed_steps, WFT_NR_smoothed_mean_rewards, color='steelblue',label='Default hyperparameters')
    plt.fill_between(WFT_NR_smoothed_steps, WFT_NR_smoothed_mean_rewards - WFT_NR_smoothed_std_rewards, WFT_NR_smoothed_mean_rewards + WFT_NR_smoothed_std_rewards, color='steelblue', alpha=0.2)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.xlabel('Time Steps', labelpad=15)
    plt.ylabel('Episode Mean Reward', labelpad=15)
    plt.legend(bbox_to_anchor=(0, 1, 1, 0.25), loc="lower right", borderaxespad=0.5, ncol=4)
    plt.show()
    # plt.grid(True)

