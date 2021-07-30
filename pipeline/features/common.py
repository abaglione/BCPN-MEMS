def train_test_split(features, train_ratio):
    # split into train and test sets and return these
    n_samples = len(features)
    train = features[0:int(n_samples*train_ratio)].values
    test = features[int(n_samples*train_ratio):].values
    return train, test