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


def synthetic_dataset_generator(
        min_samples: int,
        max_samples: int,
        min_features: int,
        max_features: int,
        max_classes: int
    ):

    config = get_prior_config_causal(max_features=max_features)

    config['prior_type'] = 'mlp'
    config['differentiable'] = True
    config['flexible'] = True

    config['num_classes'] = uniform_int_sampler_f(2, max_classes)
    config['balanced'] = False

    config['bptt_extra_samples'] = None
    config['num_features_used'] = {
        'uniform_int_sampler_f(3, max_features)': uniform_int_sampler_f(min_features, max_features)
    }
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

    config['random_feature_rotation'] = True
    config['rotate_normalized_labels'] = True

    config["mix_activations"] = False # False heisst eig True

    config['emsize'] = 512
    config['nhead'] = config['emsize'] // 128
    config['bptt'] = max_samples
    config['canonical_y_encoder'] = False

    config['aggregate_k_gradients'] = 1
    config['batch_size'] = 1
    config['num_steps'] = 2**63
    config['epochs'] = 400
    config['total_available_time_in_s'] = None #60*60*22 # 22 hours for some safety...

    config['train_mixed_precision'] = True
    config['efficient_eval_masking'] = True

    config['normalize_by_used_features'] = False

    config['min_eval_pos'] = max_samples // 2
    config['max_eval_pos'] = max_samples // 2 + 1


    config_sample = evaluate_hypers(config)

    config_sample['batch_size'] = 1

    print("IGNORE THE TRANSFORMER PARAMETER TEXT BELOW, ITS FROM LOADING TABPFN DATA GENERATOR")
    model = get_model(config_sample, 'cpu', should_train=False, verbose=0)
    print("IGNORE THE TRANSFORMER PARAMETER TEXT ABOVE, ITS FROM LOADING TABPFN DATA GENERATOR")

    data_iter = iter(model[3])

    for (_, data, _), targets, _ in data_iter:
     
        x = data[:, 0, :]
        y = targets

        if torch.all(y == -100):
            # in case of too many classes, the synthetic generator is not able to split the dataset
            # in a way that the training and validation set have the same number of classes
            # the generator returns -100 as a label for all observations in this case
            continue

        # remove all zero columns
        x = x[:, x.sum(dim=0) != 0]

        curr_samples = uniform_int_sampler_f(min_samples, max_samples)()
        x = x[:curr_samples, :]
        y = y[:curr_samples, :]

        yield x, y



if __name__  == '__main__':

    gen = synthetic_dataset_generator(
        min_samples = 100,
        max_samples = 10000,
        min_features = 3,
        max_features = 133,
        max_classes = 10
    )
    x, y = next(gen)
    pass