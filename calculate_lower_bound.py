import numpy as np
import pandas as pd
import os
import sys
import time
import json
import torch
from tqdm import tqdm
from get_data import *
from clustering_data import *
from finetune_classifiers import *
from losses import *
from config import *
from joblib import Parallel, delayed
import multiprocessing
cfg = get_cfg_defaults()

def calculate_diff_loss(df_small, df_syn, data):
    result = {}
    for name, cls_model in  tqdm(classifiers.items()):
        print(f"Classifier: {name}")
        # Pre-allocate lists for better performance
        ids = []
        areas = []
        loss_syn = []
        diff_losses = []
        diff_targets = []
        
        # Get the maximum valid index for both dataframes
        max_valid_idx = min(len(df_small), len(df_syn))
        print(f"Maximum valid index: {max_valid_idx}")
        
        for area in tqdm(data.keys()):
            # Skip if area is larger than df_small size
            if area >= len(df_small):
                print(f"Warning: Area {area} is out of bounds for df_small, skipping...")
                continue
                
            for idx in data[area]:
                # Skip if index is not in df_syn's index
                if idx not in df_syn.index:
                    print(f"Warning: Index {idx} not found in df_syn, skipping...")
                    continue
                
                try:
                    small_loss = df_small.iloc[area][name]
                    syn_loss = df_syn.loc[idx][name]  # Use loc to access by original index
                    
                    diff_loss = np.abs(small_loss - syn_loss)
                    diff_target = -(syn_loss - diff_loss)
                    
                    ids.append(idx)
                    areas.append(area)
                    diff_losses.append(diff_loss)
                    loss_syn.append(syn_loss)
                    diff_targets.append(diff_target)
                except Exception as e:
                    print(f"Error processing area {area}, index {idx}: {str(e)}")
                    continue
        
        # Create DataFrame from lists
        diff_df = pd.DataFrame({
            'id': ids,
            'area': areas,
            'loss_0_1': loss_syn,
            'diff_loss': diff_losses,
            'diff_target': diff_targets
        })
        
        # if len(diff_df) > 0:
        #     diff_df.to_csv(os.path.join(cfg.CLUSTER.SAVE_DIR_SYN_DATA, f'diff_{name}.csv'), index=False)
        #     print(f"Saved {len(diff_df)} rows to diff_{name}.csv")
        # else:
        #     print(f"Warning: No valid data for classifier {name}")
        result[name] = diff_df
    return result

def calculate_mean_max_loss_per_area(loss_df, num_clusters=cfg.DATA.TOTAL_PARTITONS):
    # print(f"Input loss_df shape: {loss_df.shape}")
    # print(f"Input loss_df columns: {loss_df.columns.tolist()}")

    result = {}
    # Get the loss DataFrame for this classifier
    loss_area_L01 = loss_df['loss_0_1']
    # print(f"Loss data type: {loss_area_L01.dtype}")
    
    mean_loss_area = np.zeros(num_clusters)
    max_loss_area = np.zeros(num_clusters)
    
    # Group by area to get losses for each area
    area_groups = loss_df.groupby('area')
    # print(f"Number of unique areas: {len(area_groups)}")
    
    for area_idx in range(num_clusters):
        print(f"\nProcessing area {area_idx}")
        try:
            # Get losses for this area
            area_data = area_groups.get_group(area_idx)
            area_losses = area_data['loss_0_1'].values  # Access the loss_0_1 column directly
            
            # Convert to float and handle any non-numeric values
            try:
                area_losses = area_losses.astype(float)
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not convert area {area_idx} losses to float: {str(e)}")
                print(f"Sample of problematic values: {area_losses[:5]}")
                area_losses = np.array([0.0])  # Default to 0 if conversion fails
            
            # print(f"Number of points in area {area_idx}: {len(area_losses)}")
            
            if len(area_losses) != 0:
                # Remove any NaN values before calculating mean and max
                valid_mask = ~np.isnan(area_losses)
                area_losses = area_losses[valid_mask]
                
                if len(area_losses) > 0:
                    mean_loss_area[area_idx] = np.mean(area_losses)
                    max_loss_area[area_idx] = np.max(area_losses)
                    # print(f"Area {area_idx} - Mean loss: {mean_loss_area[area_idx]:.4f}, Max loss: {max_loss_area[area_idx]:.4f}")
                else:
                    print(f"No valid points found in area {area_idx} after removing NaN values")
                    mean_loss_area[area_idx] = 0
                    max_loss_area[area_idx] = 0
            else:
                print(f"No points found in area {area_idx}")
                mean_loss_area[area_idx] = 0
                max_loss_area[area_idx] = 0
        except KeyError:
            print(f"Area {area_idx} not found in data")
            mean_loss_area[area_idx] = 0
            max_loss_area[area_idx] = 0
        except Exception as e:
            print(f"Unexpected error processing area {area_idx}: {str(e)}")
            mean_loss_area[area_idx] = 0
            max_loss_area[area_idx] = 0
    
    # Store results for this classifier
    result['mean_loss_area'] = mean_loss_area
    result['max_loss_area'] = max_loss_area
    
    # Save results to CSV
    # df = pd.DataFrame({
    #     'mean_loss_area': mean_loss_area,
    #     'max_loss_area': max_loss_area
    # })
    # save_path = f'mean_max_loss.csv'
    # df.to_csv(save_path, index=False)
    # print(f"Saved results to {save_path}")
        
    return result

def calculate_a_hat(mean_loss_area):
    a_hat = np.max(mean_loss_area)
    return a_hat

def calculate_beta(mean_loss_area, Pg):
    beta = 2*np.sum(np.multiply(np.array(mean_loss_area)**2, Pg))
    return beta

def calculate_C_h(max_loss_area):
    C_h = np.max(max_loss_area)
    return C_h

def check_delta_condition(delta1, a_hat, beta, check_flag, g_adj_short):
    cond = 0
    if a_hat == 0 or g_adj_short == 0:
        delta1_lb = np.inf
    else:
        delta1_lb = np.exp(-g_adj_short*beta/(2*a_hat**2))
    cond = float(delta1 > delta1_lb) # Condition is True (1.0) if delta1 > delta1_lb, False (0.0) otherwise
    return cond, delta1_lb

def calculate_F_g_h(loss_df):
    loss_area_L01 = loss_df['loss_0_1']
    F_g_h = np.average(loss_area_L01)
    return F_g_h

def calculate_epsilon(loss_df):
    loss_area_L01 = loss_df['diff_loss']
    epsilon = np.average(loss_area_L01)
    return epsilon

def calculate_uncertainties(loss_df, delta1, delta2, a_hat):
    # Calculate Uncertainty 1
    uncertainty1 = 0
    area_groups = loss_df.groupby('area')
    g = len(loss_df)
    # print(f"Total points (g): {g}")
    
    # Get unique areas from the data
    unique_areas = loss_df['area'].unique()
    # print(f"Number of unique areas in data: {len(unique_areas)}")

    for area_idx in range(cfg.DATA.TOTAL_PARTITONS):
        try:
            area_data = area_groups.get_group(area_idx)
            g_i = len(area_data)
            #print(f"Area {area_idx} has {g_i} points")
            uncertainty1 += (g_i/g) ** 2
            # uncertainty1 += Ng_adj_short[area_idx] ** 2

        except KeyError:
            print(f"Area {area_idx} not found in data, skipping...")
            uncertainty1 += 0
            continue

    uncertainty1 = uncertainty1 * np.log(1/delta2) * 1/2
    uncertainty1 = np.sqrt(uncertainty1)
    # uncertainty1 = uncertainty1 * np.log(1/delta2) * 1/2 * (1/(g_adj_short)**2)
    # Calculate Uncertainty 2
    uncertainty2 = a_hat * np.log(1/delta1) * 1/g
    
    # print(f"Total points (g): {g}")
    # print(f"Uncertainty1 calculation: {uncertainty1:.6f}")
    # print(f"Uncertainty2 calculation: {uncertainty2:.6f}")
    
    return uncertainty1, uncertainty2

def calculate_lower_bound(loss_df, delta1, delta2, Pg, check_flag, num_clusters=cfg.DATA.TOTAL_PARTITONS):
    result_mean_max = calculate_mean_max_loss_per_area(loss_df, num_clusters)

    g = len(loss_df)
    a_hat = calculate_a_hat(result_mean_max['mean_loss_area'])
    beta = calculate_beta(result_mean_max['mean_loss_area'], Pg)

    C_h = calculate_C_h(result_mean_max['max_loss_area'])

    cond, delta1_lb = check_delta_condition(delta1, a_hat, beta, check_flag, g)

    F_g_h = calculate_F_g_h(loss_df)
    epsilon = calculate_epsilon(loss_df)
    uncertainty1, uncertainty2 = calculate_uncertainties(loss_df, delta1, delta2, a_hat)

    inside = F_g_h - epsilon - C_h*uncertainty1 + uncertainty2

    if cond != 1:
        lower_bound = np.inf
        print(f"Delta1 condition not met, cond: {cond}")
    else:
        if inside > 0:
            lower_bound = np.sqrt(inside)  - np.sqrt(uncertainty2)
            lower_bound = lower_bound**2
        else:
            lower_bound = np.inf
            print(f"F(G,h) smaller than Epsilon and Uncertainty_1. Gap values: {inside}")
        
    result = {
        'delta_1': delta1,
        'delta_2': delta2,
        'beta': beta,
        'cond': cond,
        'a_hat': a_hat,
        'C_h': C_h,
        'F_g_h': F_g_h,
        'epsilon': epsilon,
        'uncertainty1': uncertainty1,
        'uncertainty2': uncertainty2,
        'lower_bound': lower_bound
    }
    return result

def process_batch(batch_indices, X_original, y_original, cls_model):
    losses = np.zeros(len(batch_indices))
    for j, indices in enumerate(batch_indices):
        X_batch = X_original.iloc[indices]
        y_batch = y_original[indices]
        losses[j] = zero_one_loss_set(X_batch, y_batch, cls_model)
    return losses

def calculate_Boostrap_loss(g_adj, delta1, delta2, clf_name, losses_csv_path=cfg.LOSS.SAVE_CLUSTER):
    # Use precomputed per-record 0/1 losses for the given classifier column.
    df_losses = pd.read_csv(losses_csv_path)
    if clf_name not in df_losses.columns:
        raise ValueError(f"Classifier column '{clf_name}' not found in {losses_csv_path}")

    per_point_losses = df_losses[clf_name].to_numpy(dtype=np.float32)
    K = per_point_losses.shape[0]
    if K == 0:
        return 0.0

    # Vectorized bootstrap via Binomial(K, p_hat)
    n_boot = int(min(cfg.LOWER_BOUND.NUM_BOOTSTRAP_ITERATIONS, g_adj))
    print(f"Using {n_boot} bootstrap iterations for K={K} with precomputed losses from {losses_csv_path}")

    p_hat = float(per_point_losses.mean())
    counts = np.random.binomial(K, p_hat, size=n_boot)
    boot_means = counts.astype(np.float32) / float(K)

    delta = float(delta1) + float(delta2)
    delta = max(0.0, min(1.0, delta))
    loss = np.percentile(boot_means, delta * 100.0)
    return loss

def calculate_synthetic_data_without_optim(g_adj, encoder, gan_model, clf, num_iter=cfg.LOWER_BOUND.NUM_WO_ITERATIONS):
    L01_loss_syn = np.zeros(num_iter)
    for i in range(num_iter):
        print(f"Iteration {i+1} of {num_iter}")

        D_syn = gan_model.sample(g_adj)
        X_syn, y_syn = split_to_X_y(D_syn, type='test', encoder=encoder)
        L01 = zero_one_loss_set(X_syn, y_syn, clf)
        L01_loss_syn[i] = L01
    return L01_loss_syn

def calculate_therical_lower_bound(no_syn_adj, df_small_test, clf_name):
    theoretical_lower_bound = 0
    K = len(df_small_test)
    loss_df = df_small_test[clf_name]
    g = np.sum(no_syn_adj)
    
    for index in range(K):
        theoretical_lower_bound += no_syn_adj[index]*loss_df.iloc[index]

    theoretical_lower_bound = theoretical_lower_bound/g
    return theoretical_lower_bound
    

if __name__ == "__main__":
    no_syn_opt = load_dis_partion(save_dir=cfg.CLUSTER.SAVE_DIR_OPT_NUM_POINTS)
    no_syn_adj = adjust_g(no_syn_opt, adjust_factor=cfg.CLUSTER.ADJUST_FACTOR)

    Ng_adj_short = np.load('Ng_adj_short.npy')
    g_adj_short = np.sum(Ng_adj_short)
    check_flag = check_no_synth_points(Ng_adj_short, no_syn_adj)

    loss_df = pd.read_csv('Datasets/Optim_data/diff_Logistic Regression.csv')

    result_mean_max = calculate_mean_max_loss_per_area(loss_df, cfg.DATA.TOTAL_PARTITONS)

    Pg = load_dis_partion(save_dir=cfg.CLUSTER.SAVE_DIR_DISTRIBUTION)

    # a_hat = calculate_a_hat(result_mean_max['mean_loss_area'])
    # beta = calculate_beta(result_mean_max['mean_loss_area'], Pg)

    # C_h = calculate_C_h(result_mean_max['max_loss_area'])

    delta1 = 0.01
    delta2 = 0.2
    # cond, delta1_lb = check_delta_condition(delta1, a_hat, beta, check_flag, g_adj_short)

    # print(f"cond: {cond}")
    # print(f"delta1_lb: {delta1_lb}")

    # F_g_h = calculate_F_g_h(loss_df)
    # epsilon = calculate_epsilon(loss_df)
    # print(f"F_g_h: {F_g_h}")
    # uncertainty1, uncertainty2 = calculate_uncertainties(loss_df, delta1, delta2, a_hat)
    # print(f"uncertainty1: {uncertainty1}")
    # print(f"uncertainty2: {uncertainty2}")

    # print('--------------------------------')
    # # print(f"Mean loss area: {result_mean_max['mean_loss_area']}")
    # # print(f"Max loss area: {result_mean_max['max_loss_area']}")
    # print(f"a_hat: {a_hat}")
    # print(f"beta: {beta}")
    # print(f"C_h: {C_h}")
    # print(f"epsilon: {epsilon}")

    # print(result_mean_max.keys())
    result = calculate_lower_bound(loss_df, delta1, delta2, Pg, check_flag)
    print(result)