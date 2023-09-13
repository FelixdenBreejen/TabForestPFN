import numpy as np
import random


class HyperparameterDrawer:
    """Random or default search in WandB sweep config format"""

    def __init__(self, cfg: dict):
        
        self.search_objects = [RandomSearchObject(name, cfg) for name, cfg in cfg.items()]

    
    def draw_random_config(self):
        return self.draw_config('random')
    

    def draw_default_config(self):
        return self.draw_config('default')
    
    
    def draw_config(self, type: str):

        assert type in ['random', 'default']
        
        config = {}

        for search_object in self.search_objects:

            if type == 'default':
                config[search_object.name] = search_object.draw_default()
            else:
                config[search_object.name] = search_object.draw_random()

        return config




class RandomSearchObject:

    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.cfg = cfg


    def draw_default(self):

        assert 'default' in self.cfg or 'value' in self.cfg, f"There is no default value for this hyperparameter: {self.name}"

        if 'default' in self.cfg:
            return self.cfg['default']
        else:
            return self.cfg['value']


    def draw_random(self):

        if 'value' in self.cfg:
            return self.cfg['value']
        
        if 'probabilities' in self.cfg:
            return self.draw_probabilities()

        if 'values' in self.cfg:
            return self.draw_values()

        if 'distribution' in self.cfg:
            return self.draw_distribution()

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
        
