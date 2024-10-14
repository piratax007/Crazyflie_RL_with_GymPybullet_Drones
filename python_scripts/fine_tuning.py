import optuna
import json
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from environments.WithoutCurriculumLearning import WithoutCurriculumLearning

def optimize_ppo(trial):
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True)
    n_steps = trial.suggest_int("n_steps", 2048, 8192)
    n_epochs = trial.suggest_int("n_epochs", 1, 12)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.25)
    ent_coef = trial.suggest_float('ent_coef', 1e-8, 0.1, log=True)
    target_kl = trial.suggest_float('target_kl', 0.000001, 0.001, log=True)
    seed = trial.suggest_categorical("seed", [0, 1234, 42, 37, 100, 83, 27, 7, 9, 10])

    env = make_vec_env(WithoutCurriculumLearning, n_envs=4)

    model = PPO('MlpPolicy',
                env,
                batch_size=batch_size,
                n_steps=n_steps,
                learning_rate=learning_rate,
                ent_coef=ent_coef,
                clip_range=clip_range,
                n_epochs=n_epochs,
                target_kl=target_kl,
                seed=seed,
    )

    model.learn(total_timesteps=int(1e6))

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)

    return mean_reward

if __name__ == '__main__':
    study = optuna.create_study(
        study_name="optimization_without_curriculum_learning",
        storage="sqlite:///results/hyperparameters.db",
        direction="maximize"
    )

    study.optimize(
        optimize_ppo,
        n_trials=100,
    )

    best_hyperparams = study.best_params
    with open('best_hyperparams.json', 'w') as f:
        json.dump(best_hyperparams, f)
