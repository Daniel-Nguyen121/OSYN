import numpy as np
import pandas as pd
import torch
from torch import nn
import faiss
import os
from collections import defaultdict
import json
from tqdm import tqdm
from config import get_cfg_defaults
cfg = get_cfg_defaults()
from classifier_config import get_hyper_params_defaults
hp = get_hyper_params_defaults()

from get_data import *
from finetune_classifiers import *
from finetune_gan import *
from clustering_data import *
from calculate_lower_bound import *

import time

def run_one_epoch_lower_bound(iter, total_iter, train_encoder, cls_models_one, gan_model, eta, preprocessor_cluster, index, df_small, no_syn, no_syn_adj, Pg, delta1, delta2, file_txt):
    print('--------------------------------')
    print(f"Iteration {iter+1} of {total_iter}")
    iter_time = time.time()

    #1. Generate synthetic data
    gen_start = time.time()
    gen_df = gan_model.sample(cfg.LOWER_BOUND.NUM_SYN_POINTS_PER_EPOCH)
    print(f"Generated {cfg.LOWER_BOUND.NUM_SYN_POINTS_PER_EPOCH} synthetic data in {time.time() - gen_start:.2f} seconds")

    #2. Clustering data
    cluster_start = time.time()
    data_df, data = cluster_syn_data(gen_df, eta, preprocessor_cluster, index)
    print(f"Clustering synthetic data time: {time.time() - cluster_start:.2f} seconds")

    #3. Calculate loss values 
    X_syn, y_syn = split_to_X_y(data_df, type='test', encoder=train_encoder)

    df_syn = get_loss_0_1_small(data_df, X_syn, y_syn, cls_models_one, preprocessor_cluster, os.path.join(cfg.LOSS.SAVE_DIR, 'syn_loss_0_1.csv'))

    diff_time = time.time()
    result_diff_df = calculate_diff_loss(df_small, df_syn, data)
    print(f"Time to calculate diff loss: {time.time() - diff_time:.2f} seconds")

    #4. Choose the optimal number of points per partion
    opti_start = time.time()
    df_optim, no_syn, no_syn_adj = choose_optim_data(result_diff_df, cfg.CLUSTER.SAVE_DIR_SYN_DATA, no_syn, no_syn_adj, cls_models_one)
    print(f"Time to choose the optimal number of points per partion: {time.time() - opti_start:.2f} seconds")

    #5. Calculate the lower bound
    # Calculate shortage and save to disk
    Ng_adj_short = Ng_adj_shortage(no_syn, no_syn_adj)  
    g_adj_short = np.sum(Ng_adj_short)
    count, ratio_count,  min_shortage, max_shortage, mean_shortage = shortage_dis(no_syn, no_syn_adj)

    # Calculate lower bound
    lower_bound_start = time.time()
    check_flag = check_no_synth_points(Ng_adj_short, no_syn_adj)

    if check_flag:
        with open(file_txt, 'a') as f:
            f.write("\t\t" + "-" * 30 + "\n\n")
            f.write(f"Total synthetic data points: {g_adj_short}\n")
            f.write("Good clustering - Full partitions are covered\n")
            f.write(f"Iteration {iter+1} of {total_iter}\n")
            f.write("\t\t" + "-" * 30 + "\n\n")
        print("Good clustering - Full partitions are covered")
        print(f"Iteration {iter+1} of {total_iter}")
    else:
        with open(file_txt, 'a') as f:
            f.write("\t\t" + "-" * 30 + "\n\n")
            f.write(f"Iteration {iter+1} of {total_iter}:\n")
            f.write(f"Total synthetic data points: {g_adj_short}\n")
            f.write(f"Count of shortage areas: {count}\n")
            f.write(f"Ratio of shortage areas: {ratio_count}\n")
            f.write(f"Min shortage: {min_shortage}\n")
            f.write(f"Max shortage: {max_shortage}\n")
            f.write(f"Mean shortage: {mean_shortage}\n")
            f.write("\t\t" + "-" * 30 + "\n\n")

    result = calculate_lower_bound(df_optim, delta1, delta2, Pg, check_flag)
    print(f"Lower bound calculation time: {time.time() - lower_bound_start:.2f} 66seconds")

    #6. Return results
    print(f"Count of shortage areas: {count}")
    print(f"Ratio of shortage areas: {ratio_count}")
    print(f"Min shortage: {min_shortage}")
    print(f"Max shortage: {max_shortage}")
    print(f"Mean shortage: {mean_shortage}")

    print(result)

    print(f"Total time for iteration {iter}: {time.time() - iter_time:.2f} seconds")
    print('--------------------------------')
    return result

def prepare_data():
    # Load the data
    data_load_start = time.time()
    data_dir = 'Datasets/Bank_Marketing/300_200'

    if not os.path.exists(data_dir):
        print(f"Dataset not found in {data_dir}, creating new dataset")
        get_dataset(id=cfg.DATA.DATASET_ID, \
        num_dels=cfg.DATA.NUM_DEL_MISSING_COLS, \
        ratio=cfg.DATA.ORACLE_RATIO, \
        seed = cfg.DATA.SEED, \
        total_points=cfg.DATA.TOTAL_PARTITONS, \
        number_dominate=cfg.DATA.NUMBER_DOMINATE)
        print(f"Dataset created in {data_dir}")
    
    loaded_df_train = load_dataset(os.path.join(data_dir, 'df_train.csv'))  
    loaded_df_small = load_dataset(os.path.join(data_dir, 'df_small.csv'))
    loaded_df_oracle = load_dataset(os.path.join(data_dir, 'df_oracle.csv'))
    print(f"Data loading time: {time.time() - data_load_start:.2f} seconds")

    # Prepare the data
    prep_start = time.time()
    X_train, y_train, train_encoder = split_to_X_y(loaded_df_train, type='train')
    X_small_test, y_small_test = split_to_X_y(loaded_df_small, type='test', encoder=train_encoder)
    X_oracle, y_oracle = split_to_X_y(loaded_df_oracle, type='test', encoder=train_encoder)
    print(f"Data preparation time: {time.time() - prep_start:.2f} seconds")
    return loaded_df_train, loaded_df_small, loaded_df_oracle, X_train, y_train, X_small_test, y_small_test, X_oracle, y_oracle, train_encoder

def prepare_classifiers(X_train, y_train, X_oracle, y_oracle, classifiers):
    categorical_columns, numerical_columns = find_columns(X_train)

    # Preprocess for train ML models
    pre_processor = preprocessor_data(numerical_columns, categorical_columns)
    
    # Train ML models
    ml_start = time.time()
    cls_models = train_cls(X_train, y_train, X_oracle, y_oracle, classifiers, pre_processor)
    print(f"ML models training time: {time.time() - ml_start:.2f} seconds")   
    return pre_processor, cls_models

def prepare_gan(loaded_df_oracle):
    # Prepare the DataFrame for GAN training
    df_gan = loaded_df_oracle.reset_index(drop=True)

    # Load the GAN
    load_start = time.time()
    if not os.path.exists(cfg.GAN.SAVE_DIR):
        model = finetune_gan(df_gan)
    else:
        print("Load the GAN")
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = load_gan(cfg.GAN.SAVE_DIR, device=device_str)
    print(f"GAN loading time: {time.time() - load_start:.2f} seconds")
    #print(model)

    # Evaluate the GAN
    eval_start = time.time()
    n = len(df_gan)
    syn_wrap = model.sample(n)
    scores_wrap = eval_synthetic(df_gan, syn_wrap)
    print(f"GAN evaluation time: {time.time() - eval_start:.2f} seconds")
    print("CTGAN wrapper:", scores_wrap)
    return model

def prepare_clustering(loaded_df_oracle, loaded_df_small, model):
    # Prepare the data for clustering
    cluster_start = time.time()
    preprocessor_cluster, dim = get_transform(loaded_df_oracle)
    
    # Get the centroids
    index, centroids = get_centroids_partion(loaded_df_small, preprocessor_cluster, dim, num_centroids=cfg.CLUSTER.NUM_CENTROIDS)

    k = get_lim_clusters(percent=cfg.CLUSTER.PERCENT_LIMIT, num_centroids=cfg.CLUSTER.NUM_CENTROIDS)
    print(f"Number of nearest clusters to get the limitation of distance: {k}")

    eta = get_radius_centroids(loaded_df_small, index, preprocessor_cluster, num_centroids=cfg.CLUSTER.NUM_CENTROIDS)
    print(f"Shape of eta: {eta.shape}")
    print(f"Max radius: {np.max(eta):.2f}")
    print(f"Min radius: {np.min(eta):.2f}")
    print(f"Mean radius: {np.mean(eta):.2f}")
    print(f"Median radius: {np.median(eta):.2f}")
    print(f"Std radius: {np.std(eta):.2f}")

    no_syn = np.zeros(cfg.DATA.TOTAL_PARTITONS)

    df_small = pd.read_csv(cfg.LOSS.SAVE_CLUSTER)
    df_oracle = pd.read_csv(cfg.LOSS.SAVE_ORACLE)

    if not os.path.exists(cfg.CLUSTER.SAVE_DIR_DISTRIBUTION):
        print("Calculate the polynomial distribution")
        calculate_polynomial_distribution(cgan_model=model,
                                    preprocessor_cluster=preprocessor_cluster,
                                    index=index,
                                    total_points=cfg.CLUSTER.NUM_CENTROIDS,
                                    num_iters=cfg.CLUSTER.NUM_ITERS_CAL_DISTRIBUTION,
                                    num_points=cfg.CLUSTER.NUM_POINTS_CAL_DISTRIBUTION,
                                    save_dir=cfg.CLUSTER.SAVE_DIR_DISTRIBUTION)
    else:
        print("Load the polynomial distribution")
        Pg = load_dis_partion(save_dir=cfg.CLUSTER.SAVE_DIR_DISTRIBUTION)

    if not os.path.exists(cfg.CLUSTER.SAVE_DIR_OPT_NUM_POINTS):
        print("Calculate the optimal number of points per partion")
        get_opt_num_points_per_partion(Pg, total_syn_points=cfg.CLUSTER.OPT_NUM_POINTS,
                                    save_dir=cfg.CLUSTER.SAVE_DIR_OPT_NUM_POINTS)
    #else:
    print("Load the optimal number of points per partion")
    no_syn_opt = load_dis_partion(save_dir=cfg.CLUSTER.SAVE_DIR_OPT_NUM_POINTS)

    print("Adjust the optimal number of points per partion")
    print(f"Adjust factor: {cfg.CLUSTER.ADJUST_FACTOR}")
    no_syn_adj = adjust_g(no_syn_opt, adjust_factor=cfg.CLUSTER.ADJUST_FACTOR)
    return preprocessor_cluster, dim, index, centroids, eta, no_syn, df_small, df_oracle, Pg, no_syn_opt, no_syn_adj

def prepare_process():
    print('--------------------------------')
    print('Prepare the data')

    # Set device
    print(f"Using device: {cfg.DEVICE}")

    #Prepare the data
    loaded_df_train, loaded_df_small, loaded_df_oracle, X_train, y_train, X_small_test, y_small_test, X_oracle, y_oracle, train_encoder = prepare_data()

    #Prepare the classifiers
    pre_processor, cls_models = prepare_classifiers(X_train, y_train, X_oracle, y_oracle, classifiers)


    #Prepare the GAN
    model = prepare_gan(loaded_df_oracle)

    #Prepare the clustering
    preprocessor_cluster, dim, index, centroids, eta, no_syn, df_small, df_oracle, Pg, no_syn_opt, no_syn_adj = prepare_clustering(loaded_df_oracle, loaded_df_small, model)

    print('--------------------------------')

    return loaded_df_train, loaded_df_small, loaded_df_oracle, X_train, y_train, X_small_test, y_small_test, X_oracle, y_oracle, train_encoder, pre_processor, cls_models, model, preprocessor_cluster, dim, index, centroids, eta, no_syn, df_small, df_oracle, Pg, no_syn_opt, no_syn_adj

def run(ep=1): #Remove ep=1 to get original version 
    loaded_df_train, loaded_df_small, loaded_df_oracle, X_train, y_train, X_small_test, y_small_test, X_oracle, y_oracle, train_encoder, pre_processor, cls_models, model, preprocessor_cluster, dim, index, centroids, eta, no_syn, df_small, df_oracle, Pg, no_syn_opt, no_syn_adj = prepare_process()

    cls_name = []
    iter_ls = []
    b_value = []
    delta_1 = []
    delta_2 = []
    loss_syn = []
    epsilon = []
    uncertainty_1 = []
    uncertainty_2 = []
    lower_bound = []
    loss_wo_opt = []
    loss_boostrap = []
    loss_theory = []
    loss_oracle = []
    loss_small = []

    start_time = time.time()
    for name, clf in tqdm(cls_models.items()):
        cls_time = time.time()
        cls_models_one = {name: clf}
        print(f"### ------ Running {name} classifier ------ ###")

        # Create result file for this model
        result_file = os.path.join(cfg.CLUSTER.TEMP_DIR, f"result_{cfg.CLUSTER.ADJUST_FACTOR}_{name}.txt")
        #result_file = f"Datasets/Temp_cluster/result_{name}.txt"
        with open(result_file, 'w') as f:
            f.write(f"Results for {name} classifier\n")
            f.write("=" * 50 + "\n\n")

        print("########## 0. Calculate Oracle - Small set Loss ##########")
        oracle_loss_start = time.time()
        oracle_loss = np.mean(df_oracle[name])
        small_loss = np.mean(df_small[name])
        print(f"Oracle loss: {oracle_loss:.4f}")
        print(f"Small loss: {small_loss:.4f}")
        print(f"Time to calculate Oracle - Small set Loss: {time.time() - oracle_loss_start:.2f} seconds")
        
        for iter in range(cfg.LOWER_BOUND.NUM_ITERATIONS):
            iter_time = time.time()
            cls_name.append(name)
            iter_ls.append(iter)
            b_value.append(cfg.CLUSTER.ADJUST_FACTOR)
            loss_oracle.append(round(oracle_loss, 4))
            loss_small.append(round(small_loss, 4))

            print("########## 1. Calculate Lower bound ##########")
            result = run_one_epoch_lower_bound(iter, cfg.LOWER_BOUND.NUM_ITERATIONS, train_encoder, cls_models_one, model, eta, preprocessor_cluster, index, df_small, no_syn, no_syn_adj, Pg, cfg.LOWER_BOUND.DELTA_1, cfg.LOWER_BOUND.DELTA_2, result_file)

            delta_1.append(result['delta_1'])
            delta_2.append(result['delta_2'])
            loss_syn.append(round(result['F_g_h'], 4))
            epsilon.append(round(result['epsilon'], 4))
            uncertainty_1.append(round(result['uncertainty1'], 4))
            uncertainty_2.append(round(result['uncertainty2'], 4))
            lower_bound.append(round(result['lower_bound'], 4))

            print("########## 2. Calculate Loss without Opt Lower bound ##########")
            g_adj = np.sum(no_syn_adj)
            loss_wo_start = time.time()
            loss_wo = calculate_synthetic_data_without_optim(g_adj, train_encoder, model, clf, num_iter=cfg.LOWER_BOUND.NUM_WO_ITERATIONS)
            print(f"Loss without Opt Lower bound: {loss_wo}")
            print(f"Time to calculate Loss without Opt Lower bound: {time.time() - loss_wo_start:.2f} seconds")
            mean_loss_wo = np.mean(loss_wo)
            std_loss_wo = np.std(loss_wo)
            wr_loss_wo = f'{mean_loss_wo:.4f} ± {std_loss_wo:.4f}'
            loss_wo_opt.append(wr_loss_wo)

            print("########## 3. Calculate Loss with Boostrap ##########")
            g_adj = np.sum(no_syn_adj)
            loss_boostrap_start = time.time()
            loss_boostrap_value = calculate_Boostrap_loss(g_adj, loaded_df_small, train_encoder, cfg.LOWER_BOUND.DELTA_1, cfg.LOWER_BOUND.DELTA_2, clf)
            print(f"Loss with Boostrap: {loss_boostrap_value}")
            print(f"Time to calculate Loss with Boostrap: {time.time() - loss_boostrap_start:.2f} seconds")
            loss_boostrap.append(round(loss_boostrap_value.item(), 4))

            print("########## 4. Calculate Loss with Theory ##########")
            loss_theory_start = time.time()
            lowerbound_theory = calculate_therical_lower_bound(no_syn_adj, df_small, name)
            print(f"Loss with Theory: {lowerbound_theory}")
            print(f"Time to calculate Loss with Theory: {time.time() - loss_theory_start:.2f} seconds")

            loss_theory_value = lowerbound_theory.item() - result['uncertainty1'] + result['uncertainty2']
            if loss_theory_value < 0:
                loss_theory_value = np.inf
            else:
                loss_theory_value = (np.sqrt(loss_theory_value) - np.sqrt(result['uncertainty2']))**2
            loss_theory.append(round(loss_theory_value, 4))

            # Write results to the model-specific file
            with open(result_file, 'a') as f:
                f.write(f"Iteration {iter + 1}:\n")
                f.write(f"b value: {cfg.CLUSTER.ADJUST_FACTOR}\n")
                f.write(f"Oracle loss: {oracle_loss:.4f}\n")
                f.write(f"Small loss: {small_loss:.4f}\n")
                f.write(f"Delta_1: {result['delta_1']:.4f}\n")
                f.write(f"Delta_2: {result['delta_2']:.4f}\n")
                f.write(f"Beta: {result['beta']:.4f}\n")
                f.write(f"a_hat: {result['a_hat']:.4f}\n")
                f.write(f"C_h: {result['C_h']:.4f}\n")
                f.write(f"F(G,h): {result['F_g_h']:.4f}\n")
                f.write(f"Epsilon: {result['epsilon']:.4f}\n")
                f.write(f"Uncertainty_1: {result['uncertainty1']:.4f}\n")
                f.write(f"Uncertainty_2: {result['uncertainty2']:.4f}\n")
                f.write(f"Lower Bound: {result['lower_bound']:.4f}\n")
                f.write(f"Loss without optimization: {mean_loss_wo:.4f} ± {std_loss_wo:.4f}\n")
                f.write(f"Loss using Boostrap: {round(loss_boostrap_value.item(), 4)}\n")
                f.write(f"Loss using Theory: {round(loss_theory_value, 4)}\n")
                f.write("-" * 30 + "\n\n")

            print(f"Total time for iteration {iter}: {time.time() - iter_time:.2f} seconds")
        print(f"Total time for {name} classifier: {time.time() - cls_time:.2f} seconds")
    print(f"Total time for all classifiers: {time.time() - start_time:.2f} seconds")
    
    save_df = pd.DataFrame({
        'Model': cls_name,
        'Iteration': iter_ls,
        'b value': b_value,
        'Delta_1': delta_1,
        'Delta_2': delta_2,
        'F(G,h)': loss_syn,
        'Epsilon': epsilon,
        'Uncertainty_1': uncertainty_1,
        'Uncertainty_2': uncertainty_2,
        'Lower bound': lower_bound,
        'Loss without Opt Lower bound': loss_wo_opt,
        'Loss with Boostrap': loss_boostrap,
        'Loss with Theory': loss_theory,
        'Loss Small': loss_small,
        'Loss Oracle': loss_oracle,
    })
    print('Final results:')
    print(save_df)
    print(f"Shape of save_df: {save_df.shape}")
    ###
    #Change this code line
    #save_df.to_csv(cfg.LOWER_BOUND.SAVE_DIR, index=False)
    
    final_csv = f'Datasets/final_result_{name}_iter_{ep}.csv'
    save_df.to_csv(final_csv, index=False)
    ###
    print(f'Save the final result to {cfg.LOWER_BOUND.SAVE_DIR}')
    print('--------------------------------')
    print("Done")
    print(f"Total time for Everything: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    ###
    #Change this code line
    #run()
    
    for i in range (1,2):
        print(f"---------------- Start Run: iteration {i} ----------------")
        run(i)
        print(f"---------------- Done Run: iteration {i} ----------------")
    ###