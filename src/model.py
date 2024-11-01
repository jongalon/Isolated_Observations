# src/model.py
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2


def self_calculate_bartlett_sphericity(data):
    """
    Perform Bartlett's test of sphericity.
    
    Parameters:
    data : dataframe
        Data frame with scaled features for analysing the Bartlett's test

    Returns:
    chi_square_value : float
        The chi-squared statistic.
    p_value : float
        The associated p-value for the test.
    """

    n_samples = data.shape[0]
    corr_matrix = data.corr(method='spearman')
    # Get the number of variables (p)
    p = corr_matrix.shape[0]
    
    # Calculate the determinant of the correlation matrix
    corr_det = np.linalg.det(corr_matrix)
    
    # Calculate the chi-square statistic using the formula
    statistic = -np.log(corr_det) * (n_samples - 1 - (2 * p + 5) / 6)
    
    # Calculate degrees of freedom
    degrees_of_freedom = p * (p - 1) / 2
    
    # Calculate the p-value from the chi-square distribution
    p_value = chi2.sf(statistic, degrees_of_freedom)
    
    return statistic, p_value

def scale_features (df_reduced):

    """
    Scales numeric features in the DataFrame for clustering.
    
    Parameters:
    df_reduced (DataFrame): The reduced DataFrame with features.
    
    Returns:
    DataFrame: Scaled features with the same index as df_reduced.
    """
    # Select the numeric columns for analysis
    data_for_clustering = df_reduced.select_dtypes(include=[float])
    labels = df_reduced['Specie_ID'].tolist()
    
    #Scale data
    scaler = StandardScaler()
    only_features_scaled = scaler.fit_transform(data_for_clustering)
    only_features_scaled = pd.DataFrame(only_features_scaled, index=df_reduced.index, columns=data_for_clustering.columns)
    return only_features_scaled



def find_lonely_observations(data: pd.DataFrame, only_features_scaled: pd.DataFrame) -> tuple:
    """
    Identifies 'isolated' observations in a dataset where an observation is not adjacent to others 
    of the same species in the dendrogram order. It also finds the nearest observation of the same 
    species and calculates the distance between them.
    
    Args:
        data (pd.DataFrame): The original dataset containing at least a 'Specie_ID' column.
        only_features_scaled (pd.DataFrame): A DataFrame with scaled features corresponding to 'data'.
    
    Returns:
        tuple: A tuple containing:
            - list of tuples: (Lonely Index, Specie_ID) for lonely observations.
            - pd.DataFrame: A DataFrame containing lonely observations with their nearest same-label index 
                            and the distance to the nearest same-label observation.
    """
    # Filter the observations from the entire dataset
    data_indices = data.index
    data_features = only_features_scaled.loc[data_indices]
    
    # Initialize a DataFrame to store the results
    lonely_observations_df = pd.DataFrame(columns=['Lonely Index', 'Specie_ID', 'Nearest Same Label Index', 'Distance'])
    
    # Calculate the dendrogram for the observations in the entire dataset
    Z_data = linkage(data_features, method='ward')
    
    # Get the order of the observations in the dendrogram
    dendro = dendrogram(Z_data, labels=data_indices, no_plot=True)
    leaves_order = dendro['leaves']
    
    # Sort the observations according to the order of the leaves in the dendrogram
    data_sorted = data.iloc[leaves_order]
    
    # Initialize a list to store lonely observations
    lonely_observations = []
    
    # Iterate over the observations in the order of the dendrogram
    for i in range(len(data_sorted)):
        species = data_sorted.iloc[i]['Specie_ID']
        idx = data_sorted.index[i]
        
        # Check adjacent observations in the order of the dendrogram
        if i > 0 and i < len(data_sorted) - 1:
            prev_species = data_sorted.iloc[i - 1]['Specie_ID']
            next_species = data_sorted.iloc[i + 1]['Specie_ID']
            if species != prev_species and species != next_species:
                lonely_observations.append((idx, species))
        elif i == 0:  # First observation
            next_species = data_sorted.iloc[i + 1]['Specie_ID']
            if species != next_species:
                lonely_observations.append((idx, species))
        elif i == len(data_sorted) - 1:  # Last observation
            prev_species = data_sorted.iloc[i - 1]['Specie_ID']
            if species != prev_species:
                lonely_observations.append((idx, species))
    
    # Calculate and store information about lonely observations
    for lonely_idx, lonely_species in lonely_observations:
        # Get the index of the lonely observation in the sorted DataFrame
        lonely_sorted_idx = data_sorted.index.get_loc(lonely_idx)
        
        # Find other observations with the same species
        same_label_indices = data_sorted[data_sorted['Specie_ID'] == lonely_species].index
        same_label_indices = same_label_indices[same_label_indices != lonely_idx]
        
        # Initialize variables to track the nearest observation
        nearest_idx = None
        nearest_distance = float('inf')
        
        for same_idx in same_label_indices:
            same_sorted_idx = data_sorted.index.get_loc(same_idx)
            distance = abs(same_sorted_idx - lonely_sorted_idx) - 1  # Number of different observations in between
            
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_idx = same_idx
        
        # Store the result if a nearest observation was found
        if nearest_idx is not None:
            lonely_observations_df = pd.concat([lonely_observations_df, pd.DataFrame([{
                'Lonely Index': lonely_idx,
                'Specie_ID': lonely_species,
                'Nearest Same Label Index': nearest_idx,
                'Distance': nearest_distance
            }])], ignore_index=True)
        else:
            # If no nearest observation was found, add the lonely observation with the entire cluster size as distance
            lonely_observations_df = pd.concat([lonely_observations_df, pd.DataFrame([{
                'Lonely Index': lonely_idx,
                'Specie_ID': lonely_species,
                'Nearest Same Label Index': np.nan,
                'Distance': len(data_sorted)  # Distance set to the size of the cluster
            }])], ignore_index=True)

    return lonely_observations, lonely_observations_df





def calculate_confusion_matrix_and_metrics(modified_set, lonely_set, total_indices):
    """
    Calculates the confusion matrix and classification metrics (precision, recall, F1-score)
    for a model that detects lonely observations as potential modifications.

    Parameters:
    - modified_set (set): Set of indices representing modified observations.
    - lonely_set (set): Set of indices representing observations detected as lonely.
    - total_indices (list): List of all observation indices in the dataset.

    Returns:
    - confusion_matrix_df (pd.DataFrame): Confusion matrix as a DataFrame with labeled rows and columns.
    - classification_metrics (dict): Dictionary of precision, recall, and F1-score for each category.
    - cm (np.array): Confusion matrix array.
    - tp, fp, fn, tn (int): True Positives, False Positives, False Negatives, True Negatives.
    """
    
    # Create binary labels for y_true (whether an index was modified) and y_pred (whether it was detected as lonely)
    y_true = [1 if idx in modified_set else 0 for idx in total_indices]
    y_pred = [1 if idx in lonely_set else 0 for idx in total_indices]

    # Calculate the confusion matrix using sklearn's confusion_matrix
    tn, fp, fn, tp = sk_confusion_matrix(y_true, y_pred).ravel()
    cm = sk_confusion_matrix(y_true, y_pred)

    # Create the confusion matrix DataFrame
    confusion_matrix_df = pd.DataFrame({
        'Predicted Modified': [tp, fp],
        'Predicted Not Modified': [fn, tn]
    }, index=['Actual Modified', 'Actual Not Modified'])

    # Calculate precision, recall, and f1-score using classification_report
    #classification_metrics = classification_report(y_true, y_pred, target_names=['Not Modified', 'Modified'])
    #print(print(classification_metrics))
    classification_metrics = classification_report(y_true, y_pred, target_names=['Not Modified', 'Modified'], output_dict=True)
    #print(print(classification_metrics))
    return confusion_matrix_df, classification_metrics, cm, tp, fp, fn, tn