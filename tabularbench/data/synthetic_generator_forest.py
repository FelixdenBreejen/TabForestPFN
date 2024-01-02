import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import QuantileTransformer
from tqdm import tqdm


def synthetic_dataset_function_forest(
        min_features = 3,
        max_features = 100,
        n_samples = 10000,
        max_classes = 10,
        base_size = 1000,
        n_estimators = 1,
        min_depth = 15,
        max_depth = 25,
    ):

    n_classes = np.random.randint(2, max_classes, size=1).item()
    categorical_perc = np.random.uniform(0, 1, size=(1,)).item()

    if min_depth == max_depth:
        depth = min_depth
    else:
        depth = np.random.randint(min_depth, max_depth, size=1).item()
    
    if min_features == max_features:
        n_features = min_features
    else:
        n_features = np.random.randint(min_features, max_features, size=1).item()

    n_categorical_features = int(categorical_perc * (n_features + 1))
    n_categorical_classes = np.random.geometric(p=0.5, size=(n_categorical_features,)) + 1

    x = np.random.normal(size=(base_size, n_features))
    y = np.random.normal(0, 1, size=(base_size,))

    clf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=depth
    )
    clf.fit(x, y)

    x2 = np.random.normal(size=(n_samples, n_features))

    if n_categorical_features > 0:
        x2_categorical = x2[:, :n_categorical_features]
        x2_numerical = x2[:, n_categorical_features:]

        quantile_transformer = QuantileTransformer(output_distribution='uniform')
        x2_categorical = quantile_transformer.fit_transform(x2_categorical)

        for i in range(n_categorical_features):
            n_categorical_class = n_categorical_classes[i]
            buckets = np.random.uniform(0, 1, size=(n_categorical_class-1,))
            buckets.sort()
            buckets = np.hstack([buckets, 1])
            b = np.argmax(x2_categorical[:, i] < buckets[:, None], axis=0)
            x2_categorical[:, i] = b

        x2 = np.hstack([x2_categorical, x2_numerical])

    z = clf.predict(x2)

    quantile_transformer = QuantileTransformer()
    z = quantile_transformer.fit_transform(z.reshape(-1, 1)).flatten()

    buckets = np.random.uniform(0, 1, size=(n_classes-1,))
    buckets.sort()
    buckets = np.hstack([buckets, 1])
    b = np.argmax(z < buckets[:, None], axis=0)

    return x2, b


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

    generator = synthetic_dataset_generator_forest(
        min_features = 3,
        max_features = 100,
        n_samples = 10000,
        max_classes = 10,
        base_size = 10000,
        n_estimators = 1,
        min_depth = 15,
        max_depth = 25,
    )

    for _ in tqdm(range(200)):        
        x, y = next(generator)