from pathlib import Path
from functools import partial

import torch

import tabularbench.models.tabPFN.encoders as encoders
from tabularbench.models.tabPFN.transformer import TransformerModel



def load_pretrained_model():
    """
    Loads a saved model from the specified position. This function only restores inference capabilities and
    cannot be used for further training.
    """

    model_state, optimizer_state, config_sample = torch.load('tabularbench/models/tabPFN/prior_diff_real_checkpoint_n_0_epoch_42.cpkt', map_location='cpu')

    if (('nan_prob_no_reason' in config_sample and config_sample['nan_prob_no_reason'] > 0.0) or
        ('nan_prob_a_reason' in config_sample and config_sample['nan_prob_a_reason'] > 0.0) or
        ('nan_prob_unknown_reason' in config_sample and config_sample['nan_prob_unknown_reason'] > 0.0)):
        encoder = encoders.NanHandlingEncoder
    else:
        encoder = partial(encoders.Linear, replace_nan_by_zero=True)

    n_out = config_sample['max_num_classes']

    encoder = encoder(config_sample['num_features'], config_sample['emsize'])

    nhid = config_sample['emsize'] * config_sample['nhid_factor']
    y_encoder_generator = encoders.get_Canonical(config_sample['max_num_classes']) \
        if config_sample.get('canonical_y_encoder', False) else encoders.Linear

    assert config_sample['max_num_classes'] > 2
    loss = torch.nn.CrossEntropyLoss(reduction='none', weight=torch.ones(int(config_sample['max_num_classes'])))

    model = TransformerModel(encoder, n_out, config_sample['emsize'], config_sample['nhead'], nhid,
                             config_sample['nlayers'], y_encoder=y_encoder_generator(1, config_sample['emsize']),
                             dropout=config_sample['dropout'],
                             efficient_eval_masking=config_sample['efficient_eval_masking'])

    # print(f"Using a Transformer with {sum(p.numel() for p in model.parameters()) / 1000 / 1000:.{2}f} M parameters")

    model.criterion = loss
    module_prefix = 'module.'
    model_state = {k.replace(module_prefix, ''): v for k, v in model_state.items()}
    model.load_state_dict(model_state)

    return model, config_sample # no loss measured