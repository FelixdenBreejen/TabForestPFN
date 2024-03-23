import numpy as np
import random

from tabularbench.core.enums import SearchType


class HyperparameterDrawer:
    """Random or default search in WandB sweep config format"""

    def __init__(self, cfg: dict):
        
        self.search_objects = [RandomSearchObject(name, cfg) for name, cfg in cfg.items()]
            
    
    def draw_config(self, search_type: SearchType) -> dict:
        
        config = {}

        for search_object in self.search_objects:

            match search_type:
                case SearchType.DEFAULT:
                    config[search_object.name] = search_object.draw_default()
                case SearchType.RANDOM:
                    config[search_object.name] = search_object.draw_random()

        for k, v, in config.items():
            if isinstance(v, float):
                config[k] = float(v)

        return config




class RandomSearchObject:

    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.cfg = cfg


    def draw_default(self):

        match self.cfg:
            case {'default': value}:
                return value
            case {'value': value}:
                return value
            case value if isinstance(value, (float, int, str, bool)):
                return value
            case _:
                raise ValueError(f'Invalid default search object?: {self.cfg}')


    def draw_random(self):

        match self.cfg:
            case {'value': value}:
                return value
            case {'probabilities': _ }:
                return self.draw_probabilities()
            case {'values': _ }:
                return self.draw_values()
            case {'distribution': _ }:
                return self.draw_distribution()
            case value if isinstance(value, (float, int, str, bool)):
                return value
            case _:
                raise ValueError(f'Invalid random search object?: {self.cfg}')
    

    def draw_probabilities(self):

        probabilities = self.cfg['probabilities']
        values = self.cfg['values']

        return random.choices(values, weights=probabilities, k=1)[0]
    

    def draw_values(self):

        values = self.cfg['values']

        return random.choice(values)
    

    def draw_distribution(self):

        if self.cfg['distribution'] == 'int_uniform':
            return self.draw_int_uniform()
        
        if self.cfg['distribution'] == 'uniform':
            return self.draw_float_uniform()
        
        if self.cfg['distribution'] == 'q_uniform':
            return self.draw_q_uniform()

        if self.cfg['distribution'] == 'log_uniform':
            return self.draw_log_uniform()

        if self.cfg['distribution'] == 'log_uniform_values':
            return self.draw_log_uniform_values()
        
        if self.cfg['distribution'] == 'q_log_uniform_values':
            return self.draw_q_log_uniform_values()
        
        if self.cfg['distribution'] == 'normal':
            return self.draw_normal()
        
        if self.cfg['distribution'] == 'q_normal':
            return self.draw_q_normal()
        
        if self.cfg['distribution'] == 'log_normal':
            return self.draw_log_normal()
        
        if self.cfg['distribution'] == 'q_log_normal':
            return self.draw_q_log_normal()
        
        raise ValueError(f'Distribution not implemented: {self.cfg["distribution"]}')


    def draw_int_uniform(self):
        assert 'min' in self.cfg and 'max' in self.cfg

        return np.random.randint(self.cfg['min'], self.cfg['max'] + 1)
    

    def draw_float_uniform(self):
        assert 'min' in self.cfg and 'max' in self.cfg

        return np.random.uniform(self.cfg['min'], self.cfg['max'])
    

    def draw_q_uniform(self):
        assert 'min' in self.cfg and 'max' in self.cfg
        x = self.draw_float_uniform()
        return self.apply_quantization(x)
    

    def draw_log_uniform(self):
        # Bit of an unintuitive function, maybe this shouldn't be used, use log_uniform_values instead
        assert 'min' in self.cfg and 'max' in self.cfg

        return np.random.uniform(np.exp(self.cfg['min']), np.exp(self.cfg['max']))


    def draw_log_uniform_values(self):
        assert 'min' in self.cfg and 'max' in self.cfg

        return np.exp(np.random.uniform(np.log(self.cfg['min']), np.log(self.cfg['max'])))
    

    def draw_q_log_uniform_values(self):
        assert 'min' in self.cfg and 'max' in self.cfg
        x = self.draw_log_uniform_values()
        return self.apply_quantization(x)
    

    def draw_normal(self):
        assert 'mu' in self.cfg and 'sigma' in self.cfg

        return np.random.normal(self.cfg['mu'], self.cfg['sigma'])
    

    def draw_q_normal(self):
        assert 'mu' in self.cfg and 'sigma' in self.cfg
        x = self.draw_normal()
        return self.apply_quantization(x)
    

    def draw_log_normal(self):
        assert 'mu' in self.cfg and 'sigma' in self.cfg

        return np.random.lognormal(self.cfg['mu'], self.cfg['sigma'])
    

    def draw_q_log_normal(self):
        assert 'mu' in self.cfg and 'sigma' in self.cfg
        x = self.draw_log_normal()
        return self.apply_quantization(x)


    def apply_quantization(self, x: float):

        if 'q' in self.cfg:
            q = self.cfg['q']
        else:
            q = 1

        return round(x / q) * q
        
