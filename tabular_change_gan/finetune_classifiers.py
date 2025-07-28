import numpy as np
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
import torch
from torch import nn
import pandas as pd
from tqdm import tqdm

from build_models import *
from get_data import *
from losses import *
from clustering_data import *

classifiers = {
     #"Nearest Neighbors": get_knn_classifier(n_neighbors=5),
     #"Linear SVM": get_svm_classifier(),
     #"Decision Tree": get_dt_classifier(),
     #"Random Forest": get_rf_classifier(max_depth=5, n_estimators=10, max_features=1),
     "Neural Net": get_mlp_classifier(),
     #"Logistic Regression": get_lr_classifier(),
     #"Gradient Boosting": get_gb_classifier(learning_rate=0.1, max_depth=3),
     #"Linear Discrimiant": get_lda_classifier(),
}

loss_names = [
    'MAE',
    'MSE',
    'JS',
    'Zero-One'
]
    
def train_cls(X_train, y_train, X_test, y_test, classifiers, preprocessor):
    cls_models = {}
    for name, clf in classifiers.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', clf)])

        pipeline.fit(X_train, y_train)
        cls_models[name] = pipeline

        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name}: Accuracy = {accuracy:.4f}")

    return cls_models

def get_loss_0_1_oracle(X_small_test, y_small_test, X_oracle, y_oracle, cls_models):
    loss_oracle = {}
    loss_small = {}
    
    for name, clf in tqdm(cls_models.items(), desc="Calculating oracle losses"):
        loss_oracle[name] = zero_one_loss_set(X_oracle, y_oracle, clf)
        loss_small[name] = zero_one_loss_set(X_small_test, y_small_test, clf)

    return loss_oracle, loss_small

def get_loss_0_1_small(df_small_test, X_small_test, y_small_test, cls_models, preprocessor, name_file=cfg.LOSS.SAVE_CLUSTER):
    # Get loss 0-1 for each element in X_small_test and save the result to a csv file
    # name_file: name of the file to save the result
    results = {name: [] for name in cls_models.keys()}
    results['vector'] = []  # Initialize vector key
    
    # Convert data to GPU tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1024  # Adjust based on your VRAM
    
    # Process vectors in batches
    for i in tqdm(range(0, len(df_small_test), batch_size), desc="Processing vectors"):
        batch_df = df_small_test.iloc[i:i+batch_size]
        batch_vectors = [transform_row(preprocessor, batch_df.iloc[[j]]) for j in range(len(batch_df))]
        results['vector'].extend(batch_vectors)
    
    # Process classifiers in batches
    for name, clf in tqdm(cls_models.items(), desc="Processing classifiers"):
        for i in tqdm(range(0, len(X_small_test), batch_size), desc=f"Processing samples for {name}", leave=False):
            batch_X = X_small_test.iloc[i:i+batch_size]
            batch_y = y_small_test[i:i+batch_size]
            
            # Process batch
            batch_losses = []
            for j in range(len(batch_X)):
                loss = zero_one_loss_element(batch_X.iloc[[j]], batch_y[j], clf)
                batch_losses.append(loss)
            
            results[name].extend(batch_losses)
    
    # Create DataFrame with ID and model columns
    df = pd.DataFrame({
        'ID': range(len(X_small_test)),
        'vector': results['vector'],
        **{name: results[name] for name in cls_models.keys()}
    })

    df.index = X_small_test.index
    
    df.to_csv(name_file, index=False)
    return df

if __name__ == "__main__":
    import time
    import os
    from config import *
    cfg = get_cfg_defaults()

    # Set device
    print(f"Using device: {cfg.DEVICE}")

    # Load the data
    data_load_start = time.time()
    data_dir = 'Datasets/Bank_Marketing/300_200'

    loaded_df_train = load_dataset(os.path.join(data_dir, 'df_train.csv'))  
    loaded_df_test = load_dataset(os.path.join(data_dir, 'df_test.csv'))
    loaded_df_small = load_dataset(os.path.join(data_dir, 'df_small.csv'))
    loaded_df_oracle = load_dataset(os.path.join(data_dir, 'df_oracle.csv'))
    print(f"Data loading time: {time.time() - data_load_start:.2f} seconds")
    
    # Prepare the data
    prep_start = time.time()
    X_train, y_train, train_encoder = split_to_X_y(loaded_df_train, type='train')
    X_small_test, y_small_test = split_to_X_y(loaded_df_small, type='test', encoder=train_encoder)
    X_oracle, y_oracle = split_to_X_y(loaded_df_oracle, type='test', encoder=train_encoder)
    print(f"Data preparation time: {time.time() - prep_start:.2f} seconds")

    categorical_columns, numerical_columns = find_columns(X_train)

    # Preprocess for train ML models
    pre_processor = preprocessor_data(numerical_columns, categorical_columns)
    
    # Train ML models
    ml_start = time.time()
    cls_models = train_cls(X_train, y_train, X_oracle, y_oracle, classifiers, pre_processor)
    print(f"ML models training time: {time.time() - ml_start:.2f} seconds")

    # Prepare the data for clustering
    cluster_start = time.time()
    preprocessor_cluster, dim = get_transform(loaded_df_oracle)
    
    # Get the centroids
    index, centroids = get_centroids_partion(loaded_df_small, preprocessor_cluster, dim, num_centroids=cfg.CLUSTER.NUM_CENTROIDS)

    k = get_lim_clusters(percent=cfg.CLUSTER.PERCENT_LIMIT, num_centroids=cfg.CLUSTER.NUM_CENTROIDS)
    print(f"Number of nearest clusters to get the limitation of distance: {k}")

    eta = get_radius_centroids(loaded_df_small, index, preprocessor_cluster, num_centroids=cfg.CLUSTER.NUM_CENTROIDS)
    print(f"Shape of eta: {eta.shape}")
    print(f"Max radius: {np.max(eta)}")
    print(f"Min radius: {np.min(eta)}")
    print(f"Mean radius: {np.mean(eta)}")
    print(f"Median radius: {np.median(eta)}")
    print(f"Std radius: {np.std(eta)}")

    # Get loss 0-1 for each element in X_small_test and save the result to a csv file
    loss_0_1_small_start = time.time()
    loss_0_1_small = get_loss_0_1_small(loaded_df_small, X_small_test, y_small_test, cls_models, preprocessor_cluster, 'small_loss_0_1.csv')
    print(f"Loss 0-1 small time: {time.time() - loss_0_1_small_start:.2f} seconds")

    # Get loss 0-1 for each element in X_test and save the result to a csv file
    loss_0_1_test_start = time.time()
    loss_0_1_big = get_loss_0_1_small(loaded_df_oracle, X_oracle, y_oracle, cls_models, preprocessor_cluster, 'oracle_loss_0_1.csv')
    print(f"Loss 0-1 test time: {time.time() - loss_0_1_test_start:.2f} seconds")