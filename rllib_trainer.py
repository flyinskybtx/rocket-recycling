import os
import tempfile
from datetime import datetime

import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.models import ModelCatalog
from ray.tune.logger import UnifiedLogger
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from custom_models import CustomMLPTorch
from rocket_gym_v1 import RocketEnvV1
from torchsummary import summary


def custom_log_creator(path_, str_):
    time_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logdir_prefix = f"{path_}_{time_str}"
    
    def logger_creator(config):
        if not os.path.exists(logdir_prefix):
            os.makedirs(logdir_prefix)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=path_)
        return UnifiedLogger(config, logdir, loggers=None)
    
    return logger_creator


def env_creater(env_config):
    return RocketEnvV1(env_config)


if __name__ == '__main__':
    ModelCatalog.register_custom_model("CustomMLPTorch", CustomMLPTorch)
    register_env('RocoketEnv-v1', env_creater)
    
    ray.init()
    custom_config = {'framework': 'torch', "num_workers": 10, "num_gpus": 0,
                     "disable_env_checking": True, "evaluation_num_workers": 1,
                     "evaluation_config": {"render_env": True,
                                           "explore": False},
                     "evaluation_interval": 50, "evaluation_duration": 1,
                     "evaluation_duration_unit": "episodes",
                     "batch_mode": "truncate_episodes",
                     "rollout_fragment_length": 200, "train_batch_size": 4000,
                     "model": {"custom_model": "CustomMLPTorch",
                               "custom_model_config": {
                                   "position_mapping": (5, 1.0),
                                   "hidden_dims": [128, 128, 128],
                                   "activation_fn": "relu",
                               }}}
    # custom_config["no_done_at_end"] = True
    # custom_config["log_level"] = 'INFO'
    # custom_config["render_env"] = True

    # config = a2c.A2C_DEFAULT_CONFIG.copy()
    config = ppo.DEFAULT_CONFIG.copy()
    config.update(custom_config)
    # algo = a2c.A2C(config, env="RocoketEnv-v1")
    algo = ppo.PPOTrainer(config, env="RocoketEnv-v1")
    
    print(algo.get_policy().model)
    
    for i in range(int(1e4)):
        results = algo.train()
        print(pretty_print(results))
        
        if i % 100 == 0:
            checkpoint = algo.save()
            print(f"Checkpoint saved at {checkpoint}")
