import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import torch

def generate_synthetic_data(test_size=0.3,
                             random_state=42):
    # Generate synthetic candidate features (e.g., 20 features)
    X, y = make_classification(n_samples=2000, n_features=20, n_informative=10,
                               n_redundant=5, n_clusters_per_class=2, random_state=random_state)
    # Simulate a sensitive attribute (e.g., gender) 
    # using the first feature
    sensitive = (X[:, 0] > 0).astype(int)
    
    # Split the dataset
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive, test_size=test_size, random_state=random_state)


    return (X_train, y_train, s_train), (X_test, y_test, s_test)


if __name__ == "__main__":
    (X_train, y_train, s_train), (X_test, y_test, s_test) = generate_synthetic_data()
    print("Data generated:")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("s_train shape:", s_train.shape)
