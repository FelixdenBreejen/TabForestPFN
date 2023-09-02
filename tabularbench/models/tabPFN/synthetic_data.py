import random
import time
import warnings
from datetime import datetime

import torch

import numpy as np

import matplotlib.pyplot as plt
from tabpfn.scripts.differentiable_pfn_evaluation import eval_model_range
from tabpfn.scripts.model_builder import get_model, get_default_spec, save_model, load_model
from tabpfn.scripts.transformer_prediction_interface import transformer_predict, get_params_from_config, load_model_workflow
from tabpfn.scripts.model_configs import *

from tabpfn.datasets import load_openml_list, open_cc_dids, open_cc_valid_dids
from tabpfn.priors.utils import plot_prior, plot_features
from tabpfn.priors.utils import uniform_int_sampler_f

from tabpfn.scripts.tabular_metrics import calculate_score_per_method, calculate_score
from tabpfn.scripts.tabular_evaluation import evaluate

from tabpfn.priors.differentiable_prior import DifferentiableHyperparameterList, draw_random_style, merge_style_with_info
from tabpfn.scripts import tabular_metrics
from tabpfn.notebook_utils import *


def synthetic_dataset_generator(n_samples, max_features):

    config = get_prior_config_causal(max_features=max_features)

    config['prior_type'] = 'mlp'
    config['differentiable'] = True
    config['flexible'] = True

    config['num_classes'] = uniform_int_sampler_f(2, config['max_num_classes'])
    config['balanced'] = False

    config['bptt_extra_samples'] = None

    # diff
    config['output_multiclass_ordered_p'] = 0.
    del config['differentiable_hyperparameters']['output_multiclass_ordered_p']

    config['multiclass_type'] = 'rank'
    del config['differentiable_hyperparameters']['multiclass_type']

    config['sampling'] = 'normal' # vielleicht schlecht?
    del config['differentiable_hyperparameters']['sampling']

    config['pre_sample_causes'] = True
    # end diff

    config['multiclass_loss_type'] = 'nono' # 'compatible'
    config['normalize_to_ranking'] = False # False

    config['categorical_feature_p'] = .2 # diff: .0

    # turn this back on in a random search!?
    config['nan_prob_no_reason'] = .0
    config['nan_prob_unknown_reason'] = .0 # diff: .0
    config['set_value_to_nan'] = .1 # diff: 1.

    config['normalize_with_sqrt'] = False

    config['new_mlp_per_example'] = True
    config['prior_mlp_scale_weights_sqrt'] = True
    config['batch_size_per_gp_sample'] = None

    config['normalize_ignore_label_too'] = False

    config['differentiable_hps_as_style'] = False
    config['max_eval_pos'] = 1000

    config['random_feature_rotation'] = True
    config['rotate_normalized_labels'] = True

    config["mix_activations"] = False # False heisst eig True

    config['emsize'] = 512
    config['nhead'] = config['emsize'] // 128
    config['bptt'] = n_samples
    config['canonical_y_encoder'] = False

        
    config['aggregate_k_gradients'] = 8
    config['batch_size'] = 8*config['aggregate_k_gradients']
    config['num_steps'] = 1024//config['aggregate_k_gradients']
    config['epochs'] = 400
    config['total_available_time_in_s'] = None #60*60*22 # 22 hours for some safety...

    config['train_mixed_precision'] = True
    config['efficient_eval_masking'] = True

    config_sample = evaluate_hypers(config)

    config_sample['batch_size'] = 4
    model = get_model(config_sample, 'cpu', should_train=False, verbose=0) # , state_dict=model[2].state_dict()
    data_iter = iter(model[3])

    for (_, data, _), targets, _ in data_iter:
     
        x = data[:, 0, :]

        # remove all zero columns
        x = x[:, x.sum(dim=0) != 0]

        y = targets

        yield x, y



if __name__  == '__main__':

    gen = synthetic_dataset_generator(n_samples=2121, max_features=133)
    x, y = next(gen)
    pass