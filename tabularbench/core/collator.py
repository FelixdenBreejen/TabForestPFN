import torch


def collate_with_padding(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:

    max_support_samples = max(dataset['x_support'].shape[0] for dataset in batch)
    max_query_samples = max(dataset['x_query'].shape[0] for dataset in batch)

    n_support_features = batch[0]['x_support'].shape[1]
    n_query_features = batch[0]['x_query'].shape[1]

    batch_size = len(batch)

    tensor_dict = {
        'x_support': torch.zeros(batch_size, max_support_samples, n_support_features, dtype=torch.float32),
        'y_support': torch.zeros(batch_size, max_support_samples, dtype=torch.int64),
        'x_query': torch.zeros(batch_size, max_query_samples, n_query_features, dtype=torch.float32),
        'y_query': torch.zeros(batch_size, max_query_samples, dtype=torch.int64),
    }

    for i, dataset in enumerate(batch):
        tensor_dict['x_support'][i, :dataset['x_support'].shape[0], :] = dataset['x_support']
        tensor_dict['y_support'][i, :dataset['y_support'].shape[0]] = dataset['y_support']
        tensor_dict['x_query'][i, :dataset['x_query'].shape[0], :] = dataset['x_query']
        tensor_dict['y_query'][i, :dataset['y_query'].shape[0]] = dataset['y_query']

    return tensor_dict

