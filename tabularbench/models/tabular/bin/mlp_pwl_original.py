# %%
import math
import scipy
import typing as ty

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import skorch


# %%
class MLP_PWL(nn.Module):
    def __init__(
        self,
        *,
        d_in: int,
        n_layers: int,
        d_layers: int, #CHANGED
        dropout: float,
        d_out: int,
        categories: ty.Optional[ty.List[int]],
        d_embedding: int,
        feature_representation_list: ty.List['FeatureRepresentation'],
        regression: bool,
        categorical_indicator
    ) -> None:
        super().__init__()

        self.regression = regression
        self.categorical_indicator = categorical_indicator #Added

        d_in = 0

        if categories is not None:
            d_in += len(categories) * d_embedding
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape=}')

        # if feature_representation_list is not None:
        #     feature_representation_list_numeric = [f for i, f in enumerate(feature_representation_list) if not categorical_indicator[i]]
        #     self.piecewiselinear = PieceWiseLinear(d_embedding, feature_representation_list_numeric, use_extra_layer=True)
        #     d_in += self.piecewiselinear.get_dim()

        # if feature_representation_list is not None:
        #     self.quantization_embedding = QuantizationEmbedding(d_embedding, feature_representation_list, use_onehot=True, use_ordinal=True, use_extra_layer=True)
        #     d_in += self.quantization_embedding.get_dim()
        
        if feature_representation_list is not None:
            if categorical_indicator is None:
                feature_representation_list_numeric = feature_representation_list
            else:
                feature_representation_list_numeric = [f for i, f in enumerate(feature_representation_list) if not categorical_indicator[i]]
            self.no_embedding = NoEmbedding(d_embedding, feature_representation_list_numeric)
            d_in += self.no_embedding.get_dim()

        d_layers = [d_layers for _ in range(n_layers)] #CHANGED

        self.layers = nn.ModuleList(
            [
                nn.Linear(d_layers[i - 1] if i else d_in, x)
                for i, x in enumerate(d_layers)
            ]
        )
        self.dropout = dropout
        self.head = nn.Linear(d_layers[-1] if d_layers else d_in, d_out)


    def forward(self, x):

        if not self.categorical_indicator is None:
            x_num = x[:, ~self.categorical_indicator].float()
            x_cat = x[:, self.categorical_indicator].long() #TODO
        else:
            x_num = x
            x_cat = None
        x_list = []
        if x_num is not None:
            x_list.append(
                torch.cat([
                    self.no_embedding(x_num),
                    # self.onehot(x_num),
                    # self.piecewiselinear_revered(x_num)
                ], dim=1)
                )
        if x_cat is not None:
            x_list.append(
                self.category_embeddings(x_cat + self.category_offsets[None]).view(
                    x_cat.size(0), -1
                )
            )
        # x_list.append(self.no_embedding(x))
        x = torch.cat(x_list, dim=-1)

        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, self.dropout, self.training)
        x = self.head(x)
        if not self.regression:
            x = x.squeeze(-1)
        return x


class QuantizationEmbedding(torch.nn.Module):

    def __init__(
        self, 
        embedding_size: int, 
        feature_representation_list: 'FeatureRepresentationList',
        use_onehot: bool,
        use_ordinal: bool,
        use_extra_layer: bool
    ):
        
        super().__init__()

        self.use_onehot = use_onehot
        self.use_ordinal = use_ordinal
        self.use_extra_layer = use_extra_layer
        self.embedding_size = embedding_size
        self.feature_representation_list = feature_representation_list

        self.compose = ComposeEmbedding(use_onehot, use_ordinal)

        self.extra_layers = torch.nn.ModuleList()

        for i, feature_representation in enumerate(feature_representation_list):
            unique_values = torch.from_numpy(feature_representation.get_bounds()).float()
            self.register_buffer(f'unique_{i}', unique_values)

            dim_in = len(unique_values) * (use_onehot + use_ordinal) + use_onehot
            extra_layer = ExtraLayer(use_extra_layer, dim_in, embedding_size)
            self.extra_layers.append(extra_layer)
            

    def forward(self, X):

        batch_size = X.shape[0]
        num_features = X.shape[1]
        x_embs = []

        for i in range(num_features):

            bounds = getattr(self, f"unique_{i}")

            lowerbound = X[:, i, None] > bounds[None, :]
    
            zeropad = torch.zeros((batch_size, 1), device=X.device).bool()
            onespad = torch.ones((batch_size, 1), device=X.device).bool()
            padded = torch.cat((onespad, lowerbound, zeropad), dim=1)

            onehot = padded[:, 1:] ^ padded[:, :-1]

            embedding = self.compose(lowerbound, onehot).float()
            embedding = self.extra_layers[i](embedding)
            
            x_embs.append(embedding)

        x = torch.cat(x_embs, dim=1)
        return x
    

    def get_dim(self):

        if self.use_extra_layer:

            return self.embedding_size * len(self.feature_representation_list)
        
        else:
            feature_representation_count = sum([len(f) for f in self.feature_representation_list])

            n_ordinal_bins = (feature_representation_count-1) * self.use_ordinal
            n_onehot_bins = feature_representation_count * self.use_onehot
            n_quantization_bins = n_ordinal_bins + n_onehot_bins

            return n_quantization_bins
        

class NoEmbedding(torch.nn.Module):

    def __init__(
        self, 
        embedding_size: int, 
        feature_representation_list: 'FeatureRepresentationList',
    ):
        
        super().__init__()

        self.embedding_size = embedding_size
        self.feature_representation_list = feature_representation_list

        self.extra_layers = torch.nn.ModuleList()

        for i, feature_representation in enumerate(feature_representation_list):
            dim_in = 1
            extra_layer = ExtraLayer(True, dim_in, embedding_size)
            self.extra_layers.append(extra_layer)
            

    def forward(self, X):

        num_features = X.shape[1]
        x_embs = []

        for i in range(num_features):
            
            embedding = self.extra_layers[i](X[:, i:i+1])
            
            x_embs.append(embedding)

        x = torch.cat(x_embs, dim=1)
        return x
    

    def get_dim(self):

        return self.embedding_size * len(self.feature_representation_list)


class ComposeEmbedding(torch.nn.Module):
    """
    This embedding selects which of the the onehot and ordinal embeddings
    should be on based on the configuration.
    """

    def __init__(self, use_onehot: bool, use_ordinal: bool):
        super().__init__()

        if use_onehot and use_ordinal:
            self.compose = CombineEmbedding()
        elif use_onehot and not use_ordinal:
            self.compose = OnehotOnly()
        elif not use_onehot and use_ordinal:
            self.compose = OrdinalOnly()
        else:
            raise ValueError("For the unique values embedding, at least onehot or ordinal needs to be on")

    def forward(self, lowerbound, onehot):
        return self.compose(lowerbound, onehot)



class CombineEmbedding(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, lowerbound, onehot):
        return torch.cat((lowerbound, onehot), dim=1)


class OrdinalOnly(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, lowerbound, onehot):
        return lowerbound


class OnehotOnly(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, lowerbound, onehot):
        return onehot



class PieceWiseLinear(torch.nn.Module):

    def __init__(self, embedding_size: int, feature_representation_list: 'FeatureRepresentationList', use_extra_layer: bool):
        super().__init__()

        self.extra_layers = torch.nn.ModuleList()
        self.use_extra_layer = use_extra_layer
        self.embedding_size = embedding_size
        self.feature_representation_list = feature_representation_list

        for i, feature_representation in enumerate(feature_representation_list):
            feature_representation_torch = torch.from_numpy(feature_representation.get_values()).float()
            self.register_buffer('feature_representation_'+str(i), feature_representation_torch)

            dim_in = len(feature_representation_torch) - 1
            extra_layer = ExtraLayer(use_extra_layer, dim_in, embedding_size)
            self.extra_layers.append(extra_layer)


    def forward(self, X_numerical):

        num_features = X_numerical.shape[1]

        newX_list = []

        for i in range(num_features):

            bounds = getattr(self, 'feature_representation_'+str(i))

            lower_bounds = bounds[:-1]
            upper_bounds = bounds[1:]

            # The following code is trying to make the following function:
            #              -----------
            #             /
            #            /
            # -----------
            # This is a piecewise linear function, where the linearly increasing part
            # is between the lower and upper bounds.
            # We can create this function with two relus.
            scaling_factor = upper_bounds - lower_bounds
            lower_part =  F.relu( X_numerical[:, None, i] - lower_bounds[None, :]) 
            upper_part =  F.relu( X_numerical[:, None, i] - upper_bounds[None, :]) 

            newX_item = (lower_part - upper_part) / scaling_factor
            newX_item = self.extra_layers[i](newX_item)

            newX_list.append(newX_item)

        newX = torch.cat(newX_list, dim=1)

        return newX
    

    def get_dim(self) -> int:

        if self.use_extra_layer:
            return len(self.feature_representation_list) * self.embedding_size
        else:
            return sum([len(f)-1 for f in self.feature_representation_list])


    

class OneHot(torch.nn.Module):

    def __init__(self, embedding_size: int, feature_representation_list: 'FeatureRepresentationList', use_extra_layer: bool):
        super().__init__()

        self.extra_layers = torch.nn.ModuleList()

        for i, feature_representation in enumerate(feature_representation_list):
            feature_representation_torch = torch.from_numpy(feature_representation.get_bounds()).float()
            self.register_buffer('feature_representation_'+str(i), feature_representation_torch)

            dim_in = len(feature_representation_torch) - 1
            extra_layer = ExtraLayer(use_extra_layer, dim_in, embedding_size)
            self.extra_layers.append(extra_layer)


    def forward(self, X_numerical):

        num_features = X_numerical.shape[1]

        newX_list = []

        for i in range(num_features):

            bounds = getattr(self, 'feature_representation_'+str(i))

            lower_bounds = bounds[:-1]
            upper_bounds = bounds[1:]

            # The following code is trying to make the following function:
            #              -----------
            #             /
            #            /
            # -----------
            # This is a piecewise linear function, where the linearly increasing part
            # is between the lower and upper bounds.
            # We can create this function with two relus.
            scaling_factor = upper_bounds - lower_bounds
            lower_part =  F.relu( X_numerical[:, None, i] - lower_bounds[None, :]) 
            upper_part =  F.relu( X_numerical[:, None, i] - upper_bounds[None, :]) 

            newX_item = (lower_part - upper_part) / scaling_factor
            newX_item = self.extra_layers[i](newX_item)

            newX_list.append(newX_item)

        newX = torch.cat(newX_list, dim=1)

        return newX
    

class ExtraLayer(torch.nn.Module):
    """
    Extra Feature-wise Layer
    """

    def __init__(self, use: bool, dim_in: int, dim_out: int):
        super().__init__()

        if use:
            self.extra_layer: torch.nn.Module = DoExtraLayer(dim_in, dim_out)
        else:
            self.extra_layer = torch.nn.Identity()

    def forward(self, X):
        return self.extra_layer(X)


class DoExtraLayer(torch.nn.Module):

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()

        self.linear = torch.nn.Linear(dim_in, dim_out)
        self.activation = torch.nn.ReLU()

    def forward(self, X):
        return self.activation(self.linear(X))



class InputShapeSetterMLP_PWL(skorch.callbacks.Callback):
    def __init__(self, regression=False, batch_size=None,
                 categorical_indicator=None, categories=None):
        self.categorical_indicator = categorical_indicator
        self.regression = regression
        self.batch_size = batch_size
        self.categories = categories


    def on_train_begin(self, net, X, y):
        print("categorical_indicator", self.categorical_indicator)

        if self.categorical_indicator is None:
            d_in = X.shape[1]
            categories = None
        else:
            d_in = X.shape[1] - sum(self.categorical_indicator)
            if self.categories is None:
                categories = list((X[:, self.categorical_indicator].max(0) + 1).astype(int))
            else:
                categories = self.categories

        max_size = net.module__d_embedding
        feature_representation_list = FeatureRepresentationList.create_representations('quantile', max_size, X)
        
        net.set_params(module__d_in=d_in,
                       module__categories=categories,  # FIXME #lib.get_categories(X_cat),
                       module__feature_representation_list=feature_representation_list,
                       module__d_out=2 if self.regression == False else 1)  # FIXME#D.info['n_classes'] if D.is_multiclass else 1,
        print("Numerical features: {}".format(d_in))
        print("Categories {}".format(categories))




class FeatureRepresentation(np.ndarray):
    """
    A feature representation is a summary of a feature in a dataset.
    It is a 1D numpy array of shape (n), where n is the size of the feature representation.
    The feature representation should have a significant smaller size than the original feature.
    Also, all values in the feature representation should be unique.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
    def get_values(self) -> np.ndarray:
        """
        Return the values of the feature representation.
        """
        return self


    def get_bounds(self, add_inf: bool = False) -> np.ndarray:
        """
        In case we have the array [1, 2, 3], we want to have the following intervals:
        (-inf, 1.5), [1.5, 2.5), [2.5, inf).
        This function returns creates the bounds without the infs: [1.5, 2.5].
        If add_inf is True, then it returns [-inf, 1.5, 2.5, inf].
        """

        right_midpoints = self[1:]
        left_midpoints = self[:-1]
        bound = (right_midpoints + left_midpoints) / 2

        if add_inf:
            bound = np.concatenate([[-np.inf], self, [np.inf]])

        return bound.view(np.ndarray)

    
    @classmethod
    def create_feature_representation_from_column(cls, repr_type: str, max_size: int, X: np.ndarray) -> 'FeatureRepresentation':
        
        if repr_type == 'quantile':
            return cls.create_quantile_representation(max_size, X)
        elif repr_type == 'unique':
            return cls.create_unique_rounded_representation(max_size, X)
        elif repr_type == 'uniform':
            return cls.create_uniform_representation(max_size, X)
        else:
            raise ValueError('Feature Representation type not supported')


    @classmethod
    def create_quantile_representation(cls, max_size: int, x: np.ndarray) -> 'FeatureRepresentation':
        
        """
        For a given dataset, gather the quantile information for each feature.
        We have n_buckets+1 values because we include the minimum and
        maximum values of each feature. For example, if n_buckets=4, then we
        have 5 values: [min, q1, q2, q3, max].
        In case of duplicate values, we remove them.
        """
        
        n_buckets = max_size
        quantiles = [i/n_buckets for i in range(n_buckets+1)]

        values = np.quantile(x, q=quantiles, interpolation='midpoint')
        values_unique = np.unique(values)

        return values_unique.view(cls)

    
    @classmethod
    def create_unique_rounded_representation(cls, max_size: int, x: np.ndarray) -> 'FeatureRepresentation':

        """
        Here we create the representation for the unique values.
        The feature representation is a 1D numpy array of shape (n), where n is the number of unique values.
        In case there are too many unique values, we round the values to a certain number of digits.
        """

        unique_values = unique_values_through_rounding(max_size, x)
        return unique_values.view(cls)
    

    @classmethod
    def create_unique_values(cls, max_size: int, x: np.ndarray) -> 'FeatureRepresentation':

        """
        Here we create the representation for the unique values.
        The feature representation is a 1D numpy array of shape (n), where n is the number of unique values.
        """

        unique_values = np.unique(x)
        return unique_values.view(cls)
    
    @classmethod
    def create_uniform_representation(cls, max_size: int, x: np.ndarray) -> 'FeatureRepresentation':

        """
        Here we create the representation for the uniform values.
        The feature representation is a 1D numpy array of shape (n), where n is the specified size.
        """

        max_element = np.max(x)
        min_element = np.min(x)
        size = max_size
        uniform_values = np.linspace(min_element, max_element, size)

        return uniform_values.view(cls)



def unique_values_through_rounding(max_size: int, features: np.ndarray) -> np.ndarray:
    """
    We want to find the number of unique values in a feature.
    In case there are too many unique values, we round the values to a certain number of digits.
    We try to pick a number of digits that results in the highest number of unique values 
    that is less than a certain threshold.
    'digits' is the number of digits we round to, but it is not necessarily an integer.
    """

    unique_values = np.unique(features)
    max_dim = max_size

    if len(unique_values) <= max_dim:
        return unique_values
       
    func = min_dist_max_dim(features, max_dim)
    digits = scipy.optimize.minimize(func, 3, method='Nelder-Mead')['x'][0]
    unique_values = get_unique_values(features, digits)

    return unique_values


def get_unique_values(features, digits):
    rounded = (features // 10**(-digits)) * 10**(-digits)
    unique_values = np.unique(rounded)
    return unique_values


def min_dist_max_dim(features, max_dim):

    def f(digits):
        unique_values = get_unique_values(features, digits)
        return (math.log(len(unique_values)) - math.log(max_dim)) ** 2

    return f




class FeatureRepresentationList(ty.List[FeatureRepresentation]):
    """
    This is a list of feature representations
    """

    def __init__(self):
        super().__init__()


    @classmethod
    def create_representations(cls, repr_type: str, max_size: int, X: np.ndarray):

        num_features = X.shape[1]
        bounds = cls()

        for i_feature in range(num_features):
            bound = FeatureRepresentation.create_feature_representation_from_column(repr_type, max_size, X[:, i_feature])
            bounds.append(bound)

        return bounds
    

    @classmethod
    def create_unique_values(cls, max_size: int, X: np.ndarray):

        num_features = X.shape[1]
        bounds = cls()

        for i_feature in range(num_features):
            bound = FeatureRepresentation.create_unique_values(max_size, X[:, i_feature])
            bounds.append(bound)

        return bounds


#
# # %%
# args, output = lib.load_config()
#
# # %%
# zero.set_randomness(args['seed'])
# dataset_dir = lib.get_path(args['data']['path'])
# stats: ty.Dict[str, ty.Any] = {
#     'dataset': dataset_dir.name,
#     'algorithm': Path(__file__).stem,
#     **lib.load_json(output / 'stats.json'),
# }
# timer = zero.Timer()
# timer.run()
#
# D = lib.Dataset.from_dir(dataset_dir)
# X = D.build_X(
#     normalization=args['data'].get('normalization'),
#     num_nan_policy='mean',
#     cat_nan_policy='new',
#     cat_policy=args['data'].get('cat_policy', 'indices'),
#     cat_min_frequency=args['data'].get('cat_min_frequency', 0.0),
#     seed=args['seed'],
# )
# if not isinstance(X, tuple):
#     X = (X, None)
#
# zero.set_randomness(args['seed'])
# Y, y_info = D.build_y(args['data'].get('y_policy'))
# lib.dump_pickle(y_info, output / 'y_info.pickle')
# X = tuple(None if x is None else lib.to_tensors(x) for x in X)
# Y = lib.to_tensors(Y)
# device = lib.get_device()
# if device.type != 'cpu':
#     X = tuple(None if x is None else {k: v.to(device) for k, v in x.items()} for x in X)
#     Y_device = {k: v.to(device) for k, v in Y.items()}
# else:
#     Y_device = Y
# X_num, X_cat = X
# if not D.is_multiclass:
#     Y_device = {k: v.float() for k, v in Y_device.items()}
#
# train_size = D.size(lib.TRAIN)
# batch_size = args['training']['batch_size']
# epoch_size = stats['epoch_size'] = math.ceil(train_size / batch_size)
#
# loss_fn = (
#     F.binary_cross_entropy_with_logits
#     if D.is_binclass
#     else F.cross_entropy
#     if D.is_multiclass
#     else F.mse_loss
# )
# args['model'].setdefault('d_embedding', None)
# model = MLP(
#     d_in=0 if X_num is None else X_num['train'].shape[1],
#     d_out=D.info['n_classes'] if D.is_multiclass else 1,
#     categories=lib.get_categories(X_cat),
#     **args['model'],
# ).to(device)
# stats['n_parameters'] = lib.get_n_parameters(model)
# optimizer = lib.make_optimizer(
#     args['training']['optimizer'],
#     model.parameters(),
#     args['training']['lr'],
#     args['training']['weight_decay'],
# )
#
# stream = zero.Stream(lib.IndexLoader(train_size, batch_size, True, device))
# progress = zero.ProgressTracker(args['training']['patience'])
# training_log = {lib.TRAIN: [], lib.VAL: [], lib.TEST: []}
# timer = zero.Timer()
# checkpoint_path = output / 'checkpoint.pt'
#
#
# def print_epoch_info():
#     print(f'\n>>> Epoch {stream.epoch} | {lib.format_seconds(timer())} | {output}')
#     print(
#         ' | '.join(
#             f'{k} = {v}'
#             for k, v in {
#                 'lr': lib.get_lr(optimizer),
#                 'batch_size': batch_size,
#                 'epoch_size': stats['epoch_size'],
#                 'n_parameters': stats['n_parameters'],
#             }.items()
#         )
#     )
#
#
# @torch.no_grad()
# def evaluate(parts):
#     model.eval()
#     metrics = {}
#     predictions = {}
#     for part in parts:
#         predictions[part] = (
#             torch.cat(
#                 [
#                     model(
#                         None if X_num is None else X_num[part][idx],
#                         None if X_cat is None else X_cat[part][idx],
#                     )
#                     for idx in lib.IndexLoader(
#                         D.size(part),
#                         args['training']['eval_batch_size'],
#                         False,
#                         device,
#                     )
#                 ]
#             )
#             .cpu()
#             .numpy()
#         )
#         try:
#             metrics[part] = lib.calculate_metrics(
#                 D.info['task_type'],
#                 Y[part].numpy(),  # type: ignore[code]
#                 predictions[part],  # type: ignore[code]
#                 'logits',
#                 y_info,
#             )
#         except ValueError as err:
#             # This happens when too deep models are applied on the Covertype dataset
#             assert (
#                 'Target scores need to be probabilities for multiclass roc_auc'
#                 in str(err)
#             )
#             metrics[part] = {'score': -999999999.0}
#     for part, part_metrics in metrics.items():
#         print(f'[{part:<5}]', lib.make_summary(part_metrics))
#     return metrics, predictions
#
#
# def save_checkpoint(final):
#     torch.save(
#         {
#             'model': model.state_dict(),
#             'optimizer': optimizer.state_dict(),
#             'stream': stream.state_dict(),
#             'random_state': zero.get_random_state(),
#             **{
#                 x: globals()[x]
#                 for x in [
#                     'progress',
#                     'stats',
#                     'timer',
#                     'training_log',
#                 ]
#             },
#         },
#         checkpoint_path,
#     )
#     lib.dump_stats(stats, output, final)
#     lib.backup_output(output)
#
#
# # %%
# timer.run()
# for epoch in stream.epochs(args['training']['n_epochs']):
#     print_epoch_info()
#
#     model.train()
#     epoch_losses = []
#     for batch_idx in epoch:
#         optimizer.zero_grad()
#         loss = loss_fn(
#             model(
#                 None if X_num is None else X_num[lib.TRAIN][batch_idx],
#                 None if X_cat is None else X_cat[lib.TRAIN][batch_idx],
#             ),
#             Y_device[lib.TRAIN][batch_idx],
#         )
#         loss.backward()
#         optimizer.step()
#         epoch_losses.append(loss.detach())
#     epoch_losses = torch.stack(epoch_losses).tolist()
#     training_log[lib.TRAIN].extend(epoch_losses)
#     print(f'[{lib.TRAIN}] loss = {round(sum(epoch_losses) / len(epoch_losses), 3)}')
#
#     metrics, predictions = evaluate([lib.VAL, lib.TEST])
#     for k, v in metrics.items():
#         training_log[k].append(v)
#     progress.update(metrics[lib.VAL]['score'])
#
#     if progress.success:
#         print('New best epoch!')
#         stats['best_epoch'] = stream.epoch
#         stats['metrics'] = metrics
#         save_checkpoint(False)
#         for k, v in predictions.items():
#             np.save(output / f'p_{k}.npy', v)
#
#     elif progress.fail:
#         break
#
#
# # %%
# print('\nRunning the final evaluation...')
# model.load_state_dict(torch.load(checkpoint_path)['model'])
# stats['metrics'], predictions = evaluate(lib.PARTS)
# for k, v in predictions.items():
#     np.save(output / f'p_{k}.npy', v)
# stats['time'] = lib.format_seconds(timer())
# save_checkpoint(True)
# print('Done!')
