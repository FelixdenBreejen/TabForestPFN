import numpy as np
from sklearn.preprocessing import QuantileTransformer


def synthetic_dataset_function_cuts(
        min_features = 3,
        max_features = 100,
        n_samples = 10000,
        max_classes = 10,
        n_cuts = 1000,
    ):

    n_classes = np.random.randint(2, max_classes, size=1).item()
    categorical_perc = np.random.uniform(0, 1, size=(1,)).item()
    
    if min_features == max_features:
        n_features = min_features
    else:
        n_features = np.random.randint(min_features, max_features, size=1).item()

    n_categorical_features = int(categorical_perc * (n_features + 1))
    n_categorical_classes = np.random.geometric(p=0.5, size=(n_categorical_features,)) + 1

    x = np.random.normal(size=(n_samples, n_features))


    if n_categorical_features > 0:
        x_categorical = x[:, :n_categorical_features]
        x_numerical = x[:, n_categorical_features:]

        quantile_transformer = QuantileTransformer(output_distribution='uniform')
        x_categorical = quantile_transformer.fit_transform(x_categorical)

        for i in range(n_categorical_features):
            n_categorical_class = n_categorical_classes[i]
            buckets = np.random.uniform(0, 1, size=(n_categorical_class-1,))
            buckets.sort()
            buckets = np.hstack([buckets, 1])
            b = np.argmax(x_categorical[:, i] < buckets[:, None], axis=0)
            x_categorical[:, i] = b

        x = np.hstack([x_categorical, x_numerical])

    A = np.random.uniform(-1, 1, size=(n_features, n_cuts))
    b = np.random.uniform(-1, 1, size=(n_cuts,))

    z = np.matmul(x, A) + b
    z = np.abs(z)

    gamma = np.random.uniform(-1, 1, size=(n_cuts, n_classes))
    y = np.matmul(z, gamma)

    y = np.argmax(y, axis=1)

    return x, y


def synthetic_dataset_generator_forest(
        min_features = 3,
        max_features = 100,
        n_samples = 10000,
        max_classes = 10,
        base_size = 1000,
        n_estimators = 1,
        min_depth = 15,
        max_depth = 25,
    ):

    while True:
        x, y = synthetic_dataset_function_forest(
            min_features = min_features,
            max_features = max_features,
            n_samples = n_samples,
            max_classes = max_classes,
            base_size = base_size,
            n_estimators = n_estimators,
            min_depth = min_depth,
            max_depth = max_depth,
        )

        yield x, y



if __name__ == '__main__':

    x, y = synthetic_dataset_function_cuts(
        min_features = 3,
        max_features = 100,
        n_samples = 10000,
        max_classes = 10,
    )

    pass

    