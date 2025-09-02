#!/usr/bin/env python3

import os
from dataclasses import dataclass
from typing import Any
import argparse
import numpy as np
import torch
from optuna import Trial, TrialPruned, Study, create_study
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv


@dataclass(frozen=True)
class StudyConfig:
    env_id: str
    total_timesteps: int
    eval_freq: int
    n_eval_episodes: int
    n_envs: int
    seed: int
    log_dir: str
    policy: str


def make_trial_directory(base_directory: str, study_id: str, trial_id: int) -> str:
    path = os.path.join(base_directory, study_id, f"trial-{trial_id}")
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "best_model"), exist_ok=True)
    return path


def make_environment(environment_id: str, n_envs: int, seed: int):
    return make_vec_env(
        environment_id,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv
    )


def sample_ppo_parameters(trial: Trial) -> dict[str, Any]:
    n_steps_exp = trial.suggest_int("n_steps_exp", 10, 14)
    batch_size_exp = trial.suggest_int("batch_size_exp", 7, 9)
    n_steps = 2 ** n_steps_exp
    batch_size = 2 ** batch_size_exp
    learning_rate = trial.suggest_float("learning_rate", 2e-5, 3e-3, log=True)
    n_epochs = trial.suggest_int("n_epochs", 5, 10)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.14, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.0001, 0.001, log=True)

    return dict(
        batch_size=batch_size,
        n_steps=n_steps,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        clip_range=clip_range,
        ent_coef=ent_coef,
    )


class PruningEvalCallback(EvalCallback):
    def __init__(
            self,
            eval_env,
            trial: Trial,
            n_eval_episodes: int,
            eval_freq: int,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=max(eval_freq, 1),
            deterministic=True,
            verbose=0
        )
        self.trial = trial
        self.eval_index = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_index += 1
            self.trial.report(self.last_mean_reward, step=self.eval_index)
            if self.trial.should_prune():
                self.is_pruned = True
                return False

        return True


def train_and_score(trial: Trial, config: StudyConfig) -> float:
    hyperparameters = sample_ppo_parameters(trial)
    working_directory = make_trial_directory(config.log_dir, trial.study.study_name, trial.number)

    train_environment = make_environment(config.env_id, config.n_envs, config.seed)
    eval_environment = make_environment(config.env_id, config.n_envs, config.seed+1)

    model = PPO(
        policy=config.policy,
        env=train_environment,
        seed=config.seed,
        verbose=0,
        tensorboard_log=working_directory,
        device='auto',
        **hyperparameters
    )

    eval_callback = PruningEvalCallback(
        eval_env=eval_environment,
        trial=trial,
        n_eval_episodes=config.n_eval_episodes,
        eval_freq=max(config.eval_freq // config.n_envs, 1)
    )

    nan_or_fail = False

    try:
        model.learn(total_timesteps=config.total_timesteps, callback=eval_callback, progress_bar=False)
    except (AssertionError, ValueError) as e:
        print(f"Trial {trial.number} failed with error {e}")
        nan_or_fail = True
    finally:
        if getattr(eval_callback, "best_mean_reward", None) is not None and np.isfinite(eval_callback.best_mean_reward):
            model.save(os.path.join(working_directory, "best", "best_model"))

        train_environment.close()
        eval_environment.close()

    if nan_or_fail:
        return float("nan")

    if eval_callback.is_pruned:
        raise TrialPruned()

    return float(eval_callback.last_mean_reward)


def build_sampler(seed: int, n_startup_trials: int, constant_liar: bool) -> TPESampler:
    return TPESampler(
        seed=seed,
        consider_prior=True,
        n_startup_trials=n_startup_trials,
        multivariate=True,
        constant_liar=constant_liar,
    )


def build_pruner(n_startup_trials: int, n_evaluations: int) -> MedianPruner:
    warmup = max(1, n_evaluations // 3)
    return MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=warmup)


def optimize(study:Study, config: StudyConfig, n_trials: int, n_jobs: int) -> None:
    def objective(trial: Trial) -> float:
        return train_and_score(trial, config)

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        gc_after_trial=True,
        show_progress_bar=False,
        catch=(TrialPruned,)
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna-SB3 PPO search for Crazyflie")
    p.add_argument("-e", "--env_id", type=str, required=True)
    p.add_argument("-s", "--study_name", type=str, default="ppo_hyperparameters")
    p.add_argument("--storage", type=str, default="sqlite://optuna_cf.db")
    p.add_argument("--total_timesteps", type=int, default=20_000_000)
    p.add_argument("--eval_freq", type=int, default=10_000)
    p.add_argument("--n_eval_episodes", type=int, default=10)
    p.add_argument("--n_trials", type=int, default=100)
    p.add_argument("--n_jobs", type=int, default=1)
    p.add_argument("--n_envs", type=int, default=1)
    p.add_argument("--seed", type=int, default=39)
    p.add_argument("--log_dir", type=str, default="results/optuna_logs")
    p.add_argument("--policy", type=str, default="MlpPolicy")
    p.add_argument("--n_startup_trials", type=int, default=10)
    p.add_argument("--n_evaluations", type=int, default=2)
    p.add_argument("--tpe_constant_liar", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch.set_num_threads(1)

    sampler = build_sampler(seed=args.seed, n_startup_trials=args.n_startup_trials, constant_liar=args.tpe_constant_liar)
    pruner = build_pruner(n_startup_trials=args.n_startup_trials, n_evaluations=args.n_evaluations)

    study = create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=args.storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    config = StudyConfig(
        env_id=args.env_id,
        total_timesteps=args.total_timesteps,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        n_envs=args.n_envs,
        seed=args.seed,
        log_dir=args.log_dir,
        policy=args.policy
    )

    optimize(study, config, n_trials=args.n_trials, n_jobs=args.n_jobs)

    best = study.best_trial
    print("\nBest value (mean eval reward):", best.value)
    print("Best hyperparameters:", best.params)
    print("resolved (powers): n_steps", best.user_attrs.get("n_steps"), "batch_size", best.user_attrs.get("batch_size"))


if __name__ == "__main__":
    main()
