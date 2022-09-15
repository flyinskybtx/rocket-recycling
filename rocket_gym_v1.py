from typing import Optional, Union, Tuple

import gym
import gym.spaces as spaces
import numpy as np
from gym.core import ObsType, ActType
from ray.tune import register_env

from rocket import Rocket


class RocketEnvV1(gym.Env):
    def __init__(self, env_config: dict):
        self.env_config = env_config
        max_steps = env_config.get('max_steps', 800)
        task = env_config.get('task', 'hover')
        rocket_type = env_config.get('rocket_type', 'falcon')
        viewport_h = env_config.get('viewport_h', 768, )
        path_to_bg_img = env_config.get('path_to_bg_img', None)
        self.rocket = Rocket(max_steps, task, rocket_type, viewport_h,
                             path_to_bg_img)
        
        high = np.array(
            [self.rocket.world_x_max, self.rocket.world_y_max,  # x, y
             np.inf, np.inf,  # vx, vy
             85 / 180 * np.pi, np.inf,  # theta, vtheta
             self.rocket.max_steps,  # t
             20 / 180 * np.pi  # phi
             ])
        low = np.array(
            [self.rocket.world_x_min, self.rocket.world_y_min,  # x, y
             -np.inf, -np.inf,  # vx, vy
             -85 / 180 * np.pi, -np.inf,  # theta, vtheta
             0,  # t
             -20 / 180 * np.pi  # phi
             ])
        self.observation_space = spaces.Box(low=low, high=high)
        self.action_space = spaces.Discrete(len(self.rocket.action_table))
    
    def reset(self, *, seed: Optional[int] = None, return_info: bool = False,
              options: Optional[dict] = None) -> Union[
        ObsType, tuple[ObsType, dict]]:
        state = self.rocket.reset()  # Flattened array from state_dict
        if not return_info:
            return state
        else:
            return state, self.rocket.state
    
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        state, reward, done, _ = self.rocket.step(
            action)  # Flattened array from state_dict
        f0, vphi0 = self.rocket.action_table[action]
        info = self.rocket.state.copy()
        info.update({'f0': f0, 'vphi0': vphi0,
                     'already_landing': self.rocket.already_landing,
                     'already_crash': self.rocket.already_crash})
        
        if self.rocket.step_id > self.rocket.max_steps:
            done = True
        
        return state, reward, done, info
    
    def render(self, mode="human"):
        self.rocket.render(window_name=f'Rocket_{self.rocket.task}')



if __name__ == '__main__':
    ENVCONFIG = {
    
    }
    rocket = RocketEnvV1(ENVCONFIG)
    state, info = rocket.reset(return_info=True)
    done = False
    while not done:
        action = rocket.action_space.sample()
        state, reward, done, info = rocket.step(action)
        rocket.render()
