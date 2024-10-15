import optuna
import json
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from environments.ObS12Stage1 import ObS12Stage1

def optimize_ppo(trial):
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 3e-4, log=True)
    n_steps = trial.suggest_int("n_steps", 512, 8192, step=256)
    n_epochs = trial.suggest_int("n_epochs", 1, 12)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.25)
    ent_coef = trial.suggest_float('ent_coef', 1e-8, 0.1, log=True)
    target_kl = trial.suggest_float('target_kl', 1e-6, 1e-3, log=True)

    env = make_vec_env(ObS12Stage1, n_envs=4)

    eval_callback = EvalCallback(
        env,
        best_model_save_path='results/hyperparams-study-log/no-normalized-reward/',
        log_path='results/hyperparams-study-log/no-normalized-reward/',
        eval_freq=int(10000 / 4),
        deterministic=True,
        render=False,
    )

    model = PPO('MlpPolicy',
                env,
                batch_size=batch_size,
                n_steps=n_steps,
                learning_rate=learning_rate,
                ent_coef=ent_coef,
                clip_range=clip_range,
                n_epochs=n_epochs,
                target_kl=target_kl,
    )

    for step in range(0, 30_000_000, 1_000_000):
        model.learn(total_timesteps=int(10e6), reset_num_timesteps=False, callback=eval_callback)

        trial.report(eval_callback.best_mean_reward, step)

        if trial.should_prune():
            raise optuna.TrialPruned()

    return eval_callback.best_mean_reward

if __name__ == '__main__':
    storage = "sqlite:///results/hyperparams-study_stage-1_no-normalized-reward.db"
    study = optuna.create_study(direction="maximize", storage=storage, study_name="ppo_hyperparams_stage-1_no-normalized-reward", load_if_exists=True)
    study.optimize(optimize_ppo, n_trials=100)

    best_hyperparams = study.best_params
    with open('best-hyperparams_Stage-1_no-normalized-reward.json', 'w') as f:
        json.dump(best_hyperparams, f)
