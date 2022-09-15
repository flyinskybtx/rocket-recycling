import ray

import os
import tempfile
from datetime import datetime

import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.evaluation import episode
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from ray.tune.logger import UnifiedLogger
from ray.tune.logger import pretty_print

from custom_models import CustomMLPTorch
from rocket_gym_v1 import RocketEnvV1

if __name__ == '__main__':
    ModelCatalog.register_custom_model("CustomMLPTorch", CustomMLPTorch)
    register_env('RocoketEnv-v1', lambda _config: RocketEnvV1(_config))

    ray.init()
    # tuner = tune.Tuner.restore(
    #     path="~/ray_results/RocketHovering"
    # )
    
    custom_config = {'framework': 'torch',
                     "num_workers": 1,
                     "num_gpus": 0,
                     "disable_env_checking": True,
                     "evaluation_num_workers": 1,
                     "evaluation_config": {"render_env": True,
                                           "explore": False},
                     "evaluation_interval": 50,
                     "evaluation_duration": 1,
                     "evaluation_duration_unit": "episodes",
                     "batch_mode": "truncate_episodes",
                     "rollout_fragment_length": 200,
                     "train_batch_size": 4000,
                     "model": {"custom_model": "CustomMLPTorch",
                               "custom_model_config": {
                                   "position_mapping": (5, 1.0),
                                   "hidden_dims": [128, 128, 128],
                                   "activation_fn": "relu",
                               }},
                     "log_level": "INFO",
                     }
    config = ppo.DEFAULT_CONFIG.copy()
    config.update(custom_config)
    # algo = a2c.A2C(config, env="RocoketEnv-v1")
    algo = ppo.PPOTrainer(config, env="RocoketEnv-v1")
    
    algo.restore("C:/Users/lxh/ray_results/RocketHovering/PPO_RocketEnv"
                 r"-v1_5ff33_00000_0_lr=0.0000_2022-09-13_17-42-26/checkpoint_000201")
    
    # --------
    env = RocketEnvV1({})
    
    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action = algo.compute_action(obs)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
        env.render()
    print("Episode reward: ", episode_reward)
