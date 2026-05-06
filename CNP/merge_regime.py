import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from itertools import combinations

# --- STEP 1: Plotting detected regimes ---
def plot_regimes(y, zt, title='Regime segmentation'):
    plt.figure(figsize=(15, 4))
    colors = plt.cm.get_cmap('tab10', np.max(zt) + 1)
    for k in np.unique(zt):
        idx = np.where(zt == k)[0]
        plt.plot(idx, y[idx], color=colors(k), label=f'Regime {k}', lw=2)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Observation")
    plt.legend()
    plt.tight_layout()
    plt.show()
###### Starting id from 0 and setting up in sequence of ascending order####    
def remap_ids(original_ids):
    
    if not isinstance(original_ids, (list, np.ndarray)):
        raise TypeError("Input 'original_ids' must be a list or a NumPy array.")

    if len(original_ids) == 0:
        return np.array([]), {}, True, False # Handle empty input

    # Convert to NumPy array for easier processing if not already
    original_ids_np = np.array(original_ids)

    # 1. Get unique sorted IDs to identify the actual IDs present
    unique_present_ids = np.sort(np.unique(original_ids_np))

    # --- Check for starting from 0 ---
    starts_from_zero = (unique_present_ids[0] == 0)
    #print(f"Original IDs start from 0: {starts_from_zero}")

    
    expected_unique_count = unique_present_ids[-1] - unique_present_ids[0] + 1
    has_skipped_ids = (len(unique_present_ids) != expected_unique_count)

    #if has_skipped_ids:
        #print("Skipped IDs (gaps) detected in original IDs.")
    #else:
        #print("No skipped IDs (gaps) detected in original IDs (within their continuous range).")

    
    old_to_new_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_present_ids)}

    remapped_ids = np.vectorize(old_to_new_id_map.get)(original_ids_np)

    return remapped_ids, old_to_new_id_map, starts_from_zero, has_skipped_ids



# --- STEP 2: Summarize statistics for each regime ---
def compute_regime_stats(y, zt):
    zt,_,_,_ = remap_ids(zt)
    stats = []
    for k in np.unique(zt):
        #print(k)
        idx = np.where(zt == k)[0]
        if len(idx) == 0: continue
        seg_y = y[idx]
        stats.append({
            'Regime': k,
            'Length': len(seg_y),
            'Mean': np.mean(seg_y),
            'Variance': np.var(seg_y),
            'Start': idx[0],
            'End': idx[-1]
        })
    return pd.DataFrame(stats)

# --- STEP 3: KL divergence between Gaussian regimes ---
def kl_divergence_gaussians(mu1, var1, mu2, var2):
    return np.log(np.sqrt(var2)/np.sqrt(var1)) + (var1 + (mu1 - mu2)**2)/(2 * var2) - 0.5

def compute_kl_matrix(stats_df):
    regimes = stats_df['Regime'].values
    K = len(regimes)
    kl_mat = np.zeros((K, K))
    for i, j in combinations(range(K), 2):
        mu1, var1 = stats_df.loc[i, 'Mean'], stats_df.loc[i, 'Variance']
        mu2, var2 = stats_df.loc[j, 'Mean'], stats_df.loc[j, 'Variance']
        kl = kl_divergence_gaussians(mu1, var1, mu2, var2)
        kl_mat[i, j] = kl_mat[j, i] = kl
    return kl_mat

def plot_kl_matrix(kl_mat):
    sns.heatmap(kl_mat, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("KL divergence between regimes")
    plt.show()

# --- STEP 4 (Optional): Merge similar regimes based on KL divergence ---
def merge_similar_regimes(zt, df_stats, kl_mat, weights_s, weights_d, threshold=0.1):
    clusters = []
    used = set()
    for i in range(len(df_stats)):
        if i in used:
            continue
        group = [i]
        #print(group)
        for j in range(len(df_stats)):
            #print(j)
            if i != j and kl_mat[i, j] < threshold:
                #print[]
                group.append(j)
                used.add(j)
                #print(group)
                #print(used)
        clusters.append(group)
        #print(clusters)
    new_weight_s = []
    new_weight_d = []
    for ll in range(len(clusters)):
        new_weight_s.append(np.sum(weights_s[clusters[ll]]))
        new_weight_d.append(np.sum(weights_d[clusters[ll]]))

    mapping = {}
    zt,_,_,_ = remap_ids(zt)
    for new_label, cluster in enumerate(clusters):
        for old_label in cluster:
            mapping[old_label] = new_label
    new_zt = np.vectorize(lambda x: mapping[x])(zt)
    return new_zt, new_weight_s, new_weight_d

