import faiss
import torch
import numpy as np
from collections import defaultdict
import json
from tqdm import tqdm
import time

from config import *
cfg = get_cfg_defaults()

from get_data import *
from finetune_classifiers import *

def get_transform(df):
    start_time = time.time()
    cat_cols, num_cols = find_columns(df)
    preprocessor_cluster = preprocessor_data(num_cols, cat_cols)
    preprocessor_cluster.fit(df)

    df_vecs = preprocessor_cluster.transform(df)
    dim = df_vecs.shape[1]
    print(f"Transform time: {time.time() - start_time:.2f} seconds")
    return preprocessor_cluster, dim

def transform_row(transform, row):
    return transform.transform(row)

def get_centroids_partion(small_df, preprocessor_cluster, dim, num_centroids=cfg.CLUSTER.NUM_CENTROIDS):
    start_time = time.time()
    print(f"Dimension of data: {dim}")
    # Use CPU version of FAISS
    index = faiss.IndexFlatL2(dim)

    centroids = []
    for i in tqdm(range(num_centroids), desc="Computing centroids"):
        row = small_df.iloc[[i]]
        vect = transform_row(preprocessor_cluster, row)
        centroids.append(vect)
    centroids = np.array(centroids)
    centroids = centroids.reshape(num_centroids, dim)
    print(f"Shape of centroids: {centroids.shape}")
    index.add(centroids)
    print(f"Centroids computation time: {time.time() - start_time:.2f} seconds")
    return index, centroids

def check_area(index, vect):
    # Find which partition that vector belongs to
    I = index.search(vect, 1)
    distance = I[0]
    cluster_id = I[1]
    return distance, cluster_id

def get_lim_clusters(percent=cfg.CLUSTER.PERCENT_LIMIT, num_centroids=cfg.CLUSTER.NUM_CENTROIDS):
    # Return number of nearest clusters to get the limitation of distance
    lim_clusters = percent/100 * num_centroids
    return int(lim_clusters)

def get_radius_centroids(df_small, index, preprocessor, percent_limit=cfg.CLUSTER.PERCENT_LIMIT, num_centroids=cfg.CLUSTER.NUM_CENTROIDS):
    start_time = time.time()
    eta = np.zeros(num_centroids)
    k = get_lim_clusters(percent=percent_limit, num_centroids=num_centroids)
    
    for i in tqdm(range(num_centroids), desc="Computing radius"):
        s = df_small.iloc[[i]]
        s = transform_row(preprocessor, s)
        I = index.search(s, k+1)
        eta[i] = np.max(I[0][0][1:])
    print(f"Radius computation time: {time.time() - start_time:.2f} seconds")
    return eta

def calculate_polynomial_distribution(cgan_model, \
 preprocessor_cluster, \
  index, \
  total_points=cfg.CLUSTER.NUM_CENTROIDS, \
  num_iters=cfg.CLUSTER.NUM_ITERS_CAL_DISTRIBUTION, \
  num_points=cfg.CLUSTER.NUM_POINTS_CAL_DISTRIBUTION, \
  save_dir=cfg.CLUSTER.SAVE_DIR_DISTRIBUTION):
    start_time = time.time()
    # raw counts of synthetic points in each of the K regions
    Ng = np.zeros(total_points)
    # empirical probability (counts / total points) per region
    Pg = np.zeros(total_points)
    # copy of Pg from the previous iteration
    check_change = np.zeros(total_points)
    # total number of synthetic points generated so far
    count_sum = 0

    # Create progress bar for iterations
    pbar = tqdm(range(num_iters), desc="Computing polynomial distribution")
    
    for no_iter in pbar:
        iter_start = time.time()
        # data_df: a df of all N synthetic points
        data_df = cgan_model.sample(num_points)
        
        # # Batch processing for faster computation
        # # Process points in batches for better performance
        # batch_size = 1000
        # for i in range(0, num_points, batch_size):
        #     end_idx = min(i + batch_size, num_points)
        #     batch_df = data_df.iloc[i:end_idx]
        #     batch_trans = preprocessor_cluster.transform(batch_df)
            
        #     # Process batch results
        #     for j in range(len(batch_trans)):
        #         dis, area = check_area(index, batch_trans[j:j+1])
        #         area = area.item()
        #         Ng[area] += 1

        for i in range(num_points):
            row = data_df.iloc[[i]]
            row_trans = transform_row(preprocessor_cluster,row)
            dis, area = check_area(index, row_trans)
            area = area.item()
            Ng[area] += 1
                
        count_sum += num_points
        Pg = Ng/count_sum

        mean_change = np.mean(np.abs(Pg - check_change))
        max_change = np.max(np.abs(Pg - check_change))

        iter_time = time.time() - iter_start
        pbar.set_postfix({
            'mean_change': f'{mean_change:.6f}',
            'max_change': f'{max_change:.6f}',
            'iter_time': f'{iter_time:.2f}s'
        })

        print(f"Iteration {no_iter}: mean change = {mean_change:.6f}, max change = {max_change:.6f}")

        check_change = Pg.copy()

    np.save(save_dir, Pg)
    total_time = time.time() - start_time
    print(f"Total distribution computation time: {total_time:.2f} seconds")
    print(f"Distribution saved in {save_dir}")

    
def load_dis_partion(save_dir=cfg.CLUSTER.SAVE_DIR_DISTRIBUTION):
    return np.load(save_dir)

def get_opt_num_points_per_partion(Pg, \
total_syn_points=cfg.CLUSTER.OPT_NUM_POINTS, \
save_dir=cfg.CLUSTER.SAVE_DIR_OPT_NUM_POINTS):
    no_syn_opt = np.random.multinomial(total_syn_points, Pg, 1)
    no_syn_opt = no_syn_opt[0]
    np.save(save_dir, no_syn_opt)
    print(f"Optimal number of points per partion saved in {save_dir}")

def compare_synDis_to_minUncer_1(syn_optimize_matrix, delta_2=cfg.LOWER_BOUND.DELTA_2):
    #Consider 0-1 loss
    C_h = 1 # C_h for L1 = 2

    uncertainty1 = 0
    num_clusters = len(syn_optimize_matrix)
    total_points = np.sum(syn_optimize_matrix)
    print(f"Total synthetic points: {total_points}")

    for area_idx in range(num_clusters):
        uncertainty1 += syn_optimize_matrix[area_idx]**2
    uncertainty1 = uncertainty1*np.log(1/delta_2)/2/total_points**2
    uncertainty1 = C_h* np.sqrt(uncertainty1) 
    min_uncertainty1 = C_h* np.sqrt(np.log(1/delta_2)/2/num_clusters)
    print(f"Uncertainty1 Opt: {uncertainty1:.4f}")
    print(f"Min Uncertainty 1: {min_uncertainty1:.4f}")

def check_no_synth_points(no_syn, no_syn_opt):
    # Check if no_syn >= no_syn_opt at each epoch
    # no_syn: So phan tu sinh tai moi vung trong lan lap thu i
    # no_syn_opt: So phan tu sinh toi uu tai moi vung
    num_clusters = len(no_syn_opt)
    check = (no_syn >= no_syn_opt)
    check = np.sum(check)
    return (check == num_clusters)


def adjust_g(no_syn, adjust_factor=cfg.CLUSTER.ADJUST_FACTOR):
    # adjust value g_i of each partion appropriate 1/K
    no_syn_adj = no_syn
    g = np.sum(no_syn_adj)
    K = len(no_syn)
    min_diff = 1/K
    small_diff = adjust_factor/K
    for i in range(K):
        if (no_syn_adj[i]/g > min_diff):
            while (no_syn_adj[i]/g-min_diff) >= small_diff:
                no_syn_adj[i] -= 1
        elif (no_syn_adj[i]/g < min_diff):
            while ((- no_syn_adj[i]/g+min_diff) >= small_diff) or (no_syn_adj[i] == 0):
                no_syn_adj[i] += 1
    return no_syn_adj

def shortage_dis(no_syn, no_syn_adj): 
    #Check about lack of data points per clusters
    # no_syn: array of length K with the 'actual' number of synthetic points, generated in each of the K partitions.
    # no_syn_adj: array of length K with the 'desired' or 'adjusted number, ideally wanted in each region (e.g. from some target distribution).
    # Turn distribution of shortage areas: how many shortage areas; /min/max/mean of insufficient no. of points
    K = len(no_syn_adj)
    shortage = np.zeros(K)
    count = 0
    for i in range(K):
        if no_syn[i] < no_syn_adj[i]:
            shortage[i] = no_syn_adj[i] - no_syn[i]
            count += 1
    return count, count/K,  np.min(shortage), np.max(shortage), np.mean(shortage)

def cluster_syn_data(df, eta, preprocessor_cluster, index, batch_size=cfg.CLUSTER.BATCH_SIZE):
    data_df = df.copy()
    N = len(data_df)
    valid_indices = set()
    valid_areas = {}  # Store area for each valid index
    data = defaultdict(list)  # Store indices for each area
    
    # Process data in batches
    for batch_start in tqdm(range(0, N, batch_size), desc="Clustering synthetic data"):
        batch_end = min(batch_start + batch_size, N)
        batch_df = data_df.iloc[batch_start:batch_end]
        
        # Transform entire batch at once
        batch_trans = preprocessor_cluster.transform(batch_df)
        
        # Get distances and areas for the entire batch
        distances, areas = index.search(batch_trans, 1)
        areas = areas.flatten()
        distances = distances.flatten()
        
        # Find valid indices in this batch
        for i, (area, dist) in enumerate(zip(areas, distances)):
            if area < len(eta) and dist <= eta[area]:
                idx = batch_start + i
                valid_indices.add(idx)
                valid_areas[idx] = area.item()  # Store the area for this index
                data[area.item()].append(idx)  # Store only the index
    
    store_time = time.time()

    # print(f"Valid indices: {valid_indices}")

    # Convert set to sorted list for consistent ordering
    valid_indices = sorted(list(valid_indices))

    # print(f"Valid indices after sorting: {valid_indices}")
    
    # Filter data_df to only keep records with valid indices but maintain original indices
    data_df = data_df.iloc[valid_indices].copy()
    # Set the index to match the original indices
    data_df.index = valid_indices

    data_df.to_csv(os.path.join(cfg.CLUSTER.TEMP_DIR, 'clustered_data.csv'), index=False)

    with open(os.path.join(cfg.CLUSTER.TEMP_DIR, 'cluster_assignments.txt'), 'w') as f:
        for area, indices in data.items():
            f.write(f"Area {area}: {indices}\n")

    print(f"Time to store data: {time.time() - store_time:.2f} seconds")
    
    return data_df, data
'''
def choose_optim_data(result_df, folder_csv, no_syn, no_syn_adj, cls_models):
    for name, cls_model in tqdm(cls_models.items()):
        current_df = result_df[name]
        df_csv = os.path.join(folder_csv, f'diff_{name}.csv')
        if os.path.exists(df_csv):
            previous_df = pd.read_csv(df_csv)
        else:
            previous_df = pd.DataFrame(columns=current_df.columns)
        
        # # Debug prints
        # print(f"\nProcessing classifier: {name}")
        # print(f"Previous DataFrame columns: {current_df.columns.tolist()}")
        # print(f"Current DataFrame columns: {previous_df.columns.tolist()}")
        
        # Ensure both DataFrames have unique indices
        current_df = current_df.reset_index(drop=True)
        previous_df = previous_df.reset_index(drop=True)
        
        # Merge the DataFrames using outer join to keep all records
        # Use suffixes to distinguish columns from each DataFrame
        merged_df = pd.merge(current_df, previous_df, 
                           on=['id', 'area', 'diff_loss', 'loss_0_1', 'diff_target'],  # Specify columns to merge on
                           how='outer',
                           suffixes=('_prev', '_curr'))
        
        # print(f"Merged DataFrame columns: {merged_df.columns.tolist()}")
        
        # Get unique areas from both DataFrames
        areas = pd.concat([current_df['area'], previous_df['area']]).unique()
        
        df_save = pd.DataFrame(columns=merged_df.columns)

        for area_idx in areas:
            print(f"\nProcessing area: {area_idx}")
            # Get points for this area from current DataFrame
            previous_area_points = previous_df[previous_df['area'] == area_idx]
            num_previous_points = len(previous_area_points)
            num_current_points = len(current_df[current_df['area'] == area_idx])

            print(f"Number of previous points in area {area_idx}: {num_previous_points}")
            print(f"Number of current points in area {area_idx}: {num_current_points}")

            if num_current_points > 0:
                # Get all points for this area from merged DataFrame
                merged_df_area = merged_df[merged_df['area'] == area_idx]
                num_total_points = len(merged_df_area)
                print(f"Total points in area {area_idx} after merge: {num_total_points}")

                # Update no_syn count
                no_syn[area_idx] += num_current_points
                
                # Get targets for this area
                targets = merged_df_area['diff_target'].tolist()

                if targets:
                    targets = np.array(targets)
                    targets = targets.reshape(-1)
                    
                    if num_total_points <= no_syn_adj[area_idx]:
                        df_optim = merged_df_area
                        print(f"Using all {num_total_points} points for area {area_idx}")
                    else:
                        if no_syn_adj[area_idx] > 0:
                            idx_min = np.argpartition(targets, no_syn_adj[area_idx])[:no_syn_adj[area_idx]]
                            df_optim = merged_df_area.iloc[idx_min]
                            print(f"Selected {len(idx_min)} points for area {area_idx}")
                        else:
                            df_optim = pd.DataFrame(columns=merged_df.columns)
                            # print(f"No points selected for area {area_idx} (no_syn_adj is 0)")
                else:
                    df_optim = pd.DataFrame(columns=merged_df.columns)
                    print(f"No targets found for area {area_idx}")
            else:
                print(f"No current points in area {area_idx}")
                continue
            
            df_save = pd.concat([df_save, df_optim])
        # Save the optimal DataFrame for this area
        df_save.to_csv(df_csv, mode='w', index=False)
        print(f"Saved optimal data for {name} in area {area_idx} to {df_csv}")

    return df_save, no_syn, no_syn_adj
'''

def choose_optim_data(result_df, folder_csv, no_syn, no_syn_adj, cls_models, b_value=cfg.CLUSTER.ADJUST_FACTOR):
    for name, cls_model in tqdm(cls_models.items()):
        current_df = result_df[name]
        df_csv = os.path.join(folder_csv, f'diff_{name}_{b_value}.csv')
        if os.path.exists(df_csv):
            previous_df = pd.read_csv(df_csv)
        else:
            previous_df = pd.DataFrame(columns=current_df.columns)
        
        # # Debug prints
        # print(f"\nProcessing classifier: {name}")
        # print(f"Previous DataFrame columns: {current_df.columns.tolist()}")
        # print(f"Current DataFrame columns: {previous_df.columns.tolist()}")
        
        # Ensure both DataFrames have unique indices
        current_df = current_df.reset_index(drop=True)
        previous_df = previous_df.reset_index(drop=True)
        
        # Merge the DataFrames using outer join to keep all records
        # Use suffixes to distinguish columns from each DataFrame
        merged_df = pd.merge(current_df, previous_df, 
                           on=['id', 'area', 'diff_loss', 'loss_0_1', 'diff_target'],  # Specify columns to merge on
                           how='outer',
                           suffixes=('_prev', '_curr'))
        
        # print(f"Merged DataFrame columns: {merged_df.columns.tolist()}")
        
        # Get unique areas from both DataFrames
        areas = pd.concat([current_df['area'], previous_df['area']]).unique()
        
        df_save = pd.DataFrame(columns=merged_df.columns)

        for area_idx in areas:
            print(f"\nProcessing area: {area_idx}")
            # Get points for this area from current DataFrame
            previous_area_points = previous_df[previous_df['area'] == area_idx]
            num_previous_points = len(previous_area_points)
            num_current_points = len(current_df[current_df['area'] == area_idx])

            print(f"Number of previous points in area {area_idx}: {num_previous_points}")
            print(f"Number of current points in area {area_idx}: {num_current_points}")

            if num_current_points > 0:
                # Get all points for this area from merged DataFrame
                merged_df_area = merged_df[merged_df['area'] == area_idx]
                num_total_points = len(merged_df_area)
                print(f"Total points in area {area_idx} after merge: {num_total_points}")

                # Update no_syn count
                no_syn[area_idx] += num_current_points
                
                # Get targets for this area
                targets = merged_df_area['diff_target'].tolist()

                if targets:
                    targets = np.array(targets)
                    targets = targets.reshape(-1)
                    
                    if num_total_points <= no_syn_adj[area_idx]:
                        df_optim = merged_df_area
                        print(f"Using all {num_total_points} points for area {area_idx}")
                    else:
                        if no_syn_adj[area_idx] > 0:
                            idx_min = np.argpartition(targets, no_syn_adj[area_idx])[:no_syn_adj[area_idx]]
                            df_optim = merged_df_area.iloc[idx_min]
                            print(f"Selected {len(idx_min)} points for area {area_idx}")
                        else:
                            df_optim = pd.DataFrame(columns=merged_df.columns)
                            # print(f"No points selected for area {area_idx} (no_syn_adj is 0)")
                else:
                    df_optim = pd.DataFrame(columns=merged_df.columns)
                    print(f"No targets found for area {area_idx}")
            else:
                print(f"No current points in area {area_idx}")
                continue
            
            df_save = pd.concat([df_save, df_optim])
        # Save the optimal DataFrame for this area
        df_save.to_csv(df_csv, mode='w', index=False)
        print(f"Saved optimal data for {name} in area {area_idx} to {df_csv}")

    return df_save, no_syn, no_syn_adj


def read_pkl(oracle_df, cls_name, file_name):
    try:
        file_path = os.path.join(cfg.CLUSTER.SAVE_DIR_SYN_DATA, cls_name, file_name)
        df = pd.read_pickle(file_path)
        return df
    except (FileNotFoundError, EOFError, pd.errors.EmptyDataError):
        df = pd.DataFrame(columns=oracle_df.columns)
        save_dir = os.path.join(cfg.CLUSTER.SAVE_DIR_SYN_DATA, cls_name)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file_name)
        df.to_pickle(file_path)
        return df



def Ng_adj_shortage(no_syn, no_syn_adj):
    # Count no. of synth points g_i each area with shortage
    g_adj_short = no_syn_adj.copy()
    K = len(no_syn_adj)
    for i in range(K):
        if (no_syn[i] < no_syn_adj[i]):
            g_adj_short[i] = no_syn[i]
    return g_adj_short

if __name__ == "__main__":
    no_syn_opt = load_dis_partion('Datasets/Distribution_data/batch_opt_syn_total_100000_partion_500.npy') 
    compare_synDis_to_minUncer_1(no_syn_opt, delta_2=0.2)
    no_syn_adj = adjust_g(no_syn_opt)
    compare_synDis_to_minUncer_1(no_syn_adj, delta_2=0.2)