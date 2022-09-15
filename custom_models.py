from typing import Dict, List

import gym
import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType

torch, nn = try_import_torch()


class PositionalMapping(nn.Module):
    """
    Positional mapping Layer.
    This layer map continuous input coordinates into a higher dimensional space
    and enable the prediction to more easily approximate a higher frequency function.
    See NERF paper for more details (https://arxiv.org/pdf/2003.08934.pdf)
    """
    
    def __init__(self, input_dim, L=5, scale=1.0):
        super(PositionalMapping, self).__init__()
        self.L = L
        self.output_dim = input_dim * (L * 2 + 1)
        self.scale = scale
    
    def forward(self, x):
        
        x = x * self.scale
        
        if self.L == 0:
            return x
        
        h = [x]
        PI = 3.1415927410125732
        for i in range(self.L):
            x_sin = torch.sin(2 ** i * PI * x)
            x_cos = torch.cos(2 ** i * PI * x)
            h.append(x_sin)
            h.append(x_cos)
        
        return torch.cat(h, dim=-1) / self.scale


class CustomMLPTorch(TorchModelV2, nn.Module):
    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space,
                              num_outputs,
                              model_config,
                              name)
        nn.Module.__init__(self)
        
        input_dim = int(np.prod(obs_space.shape))
        L, scale = tuple(model_config.get('position_mapping', (5, 1.0)))
        hidden_dims = list(model_config.get('hidden_dims', []))
        activation_fn = model_config.get('activation')
        
        # Create the layers
        self._pmap_layer = PositionalMapping(input_dim, L, scale)
        prev_size = self._pmap_layer.output_dim
        
        hidden_layers = []
        for size in hidden_dims:
            hidden_layers.append(
                SlimFC(prev_size, size, activation_fn=activation_fn,
                       initializer=normc_initializer(1.0)))
            prev_size = size
        
        self._hidden_layers = nn.Sequential(*hidden_layers)
        self._logits_layer = SlimFC(prev_size, num_outputs,
                                    initializer=normc_initializer(0.01),
                                    activation_fn=None)
        self._value_layer = SlimFC(prev_size, 1,
                                   initializer=normc_initializer(0.01),
                                   activation_fn=None, )
        
        self._pmap = None
        self._features = None
        self._last_flat_in = None
    
    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType, ) -> (TensorType, List[TensorType]):
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._pmap = self._pmap_layer(self._last_flat_in)
        self._features = self._hidden_layers(self._pmap)
        logits = self._logits_layer(self._features)
        
        return logits, state
    
    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        return self._value_layer(self._features).squeeze(1)

