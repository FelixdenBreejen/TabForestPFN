
from tabularbench.models.tabular.bin.ft_transformer import Transformer, InputShapeSetterTransformer

from tabularbench.core.trainer_old import Trainer
from tabularbench.core.trainer_pfn import TrainerPFN
from tabularbench.core.trainer_masked_saint_finetune import TrainerMaskedSaintFinetune



def modify_config(model_config, id):

    if "lr_scheduler" not in model_config:
        model_config['lr_scheduler'] = False
        
    if "es_patience" not in model_config:
        model_config['es_patience'] = 40
        
    if "lr_patience" not in model_config:
        model_config['lr_patience'] = 30
        
    if "categories" not in model_config:
        model_config['categories'] = None

    model_config['id'] = id


def create_ft_transformer_torch(model_config, id, use_checkpoints=True):

    model_config = {**model_config}
    
    modify_config(model_config, id)

    trainer = Trainer(
        Model=Transformer,
        InputShapeSetter=InputShapeSetterTransformer,
        model_config=model_config,
    )

    return trainer


def create_tab_pfn_torch(model_config, id):

    model_config = {**model_config}
    
    modify_config(model_config, id)

    trainer = TrainerPFN(model_config=model_config)

    return trainer


def create_masked_saint_torch(model_config, id):
    
    model_config = {**model_config}
    
    modify_config(model_config, id)

    trainer = TrainerMaskedSaintFinetune(model_config=model_config)

    return trainer