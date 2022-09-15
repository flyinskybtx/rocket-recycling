import ray
from ray import air, tune
from ray.rllib.models import ModelCatalog
from ray.tune import register_env

from custom_models import CustomMLPTorch
from rocket_gym_v1 import RocketEnvV1

if __name__ == '__main__':
    register_env('RocketEnv-v1', lambda _config: RocketEnvV1(_config))
    ModelCatalog.register_custom_model("CustomMLPTorch", CustomMLPTorch)
    
    ray.init()
    tuner = tune.Tuner(
        "PPO",
        tune_config=tune.TuneConfig(
            metric="episode_reward_mean",
            mode="max",
            num_samples=1,
        ),
        run_config=air.RunConfig(
            name="RocketHovering",
            stop={"training_iteration": int(1e4),
                  "episode_reward_mean": 300.0},
            checkpoint_config=air.CheckpointConfig(
                num_to_keep=5,
                checkpoint_score_attribute='episode_reward_mean',
                checkpoint_frequency=5,
                checkpoint_at_end=True,
            ),
        ),
        param_space={
            "env": "RocketEnv-v1",
            'framework': 'torch',
            "num_workers": 1,
            "num_gpus": 0,
            "disable_env_checking": True,
            "evaluation_num_workers": 1,
            "evaluation_config": {"render_env": True,
                                  "explore": False},
            "evaluation_interval": 10,
            "evaluation_duration": 1,
            "evaluation_duration_unit": "episodes",
            "batch_mode": "truncate_episodes",
            "model": {"custom_model": "CustomMLPTorch",
                      "custom_model_config": {
                          "position_mapping": (5, 1.0),
                          "hidden_dims": [128, 128, 128],
                          "activation_fn": "relu",
                      }},
            # "num_sgd_iter": tune.choice([10, 20, 30]),
            "sgd_minibatch_size": 128,  # tune.choice([128, 512, 2048])
            "rollout_fragment_length": 4000,
            "train_batch_size": 80000,  # tune.choice([10000, 20000,
            # 40000])
            "lr": tune.choice([1e-3, 1e-4, 1e-5])
        }
    )
    
    results = tuner.fit()
    print("best hyperparameters: ", results.get_best_result().config)
