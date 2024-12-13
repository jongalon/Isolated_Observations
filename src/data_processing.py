# src/data_processing.py
import pandas as pd  # Para manipular DataFrames
import numpy as np
import librosa 
from scipy import signal

def extract_features(path, start, end, sr=None, freq_range=(0, np.inf)):
    """
    Extract spectral features from an audio file within a specified frequency range.
    
    Parameters:
    path (str): Path to the audio file.
    start (float): Start time in seconds.
    end (float): End time in seconds.
    sr (int, optional): Sampling rate. If None, the default sampling rate of librosa is used.
    freq_range (tuple): A tuple (low_freq, high_freq) specifying the frequency range for feature extraction.
    
    Returns:
    dict: A dictionary with extracted spectral features within the specified frequency range.
    """
    features = {}
    try:
        # Load the audio file
        segment , sr = librosa.load(path, sr=None, offset=start, duration=end - start)
        
        # Calculate the number of samples in the signal
        num_samples = len(segment)
        if num_samples == 0:
            raise ValueError("Loaded audio signal is empty.")

        n_fft = min(2048, len(segment))
        hop_length = n_fft // 4
        f, t, Sxx = signal.spectrogram(x=segment, fs=sr, nperseg=n_fft, noverlap=hop_length)
        
        min_freq, max_freq = freq_range
        freq_mask = (f >= min_freq) & (f <= max_freq)
        Sxx_filtered = Sxx[freq_mask, :]
        
        # Calculate spectral centroid, bandwidth, and flatness
        magnitude_spectrum = np.abs(Sxx_filtered)
        spectral_centroid = np.sum(f[freq_mask, np.newaxis] * magnitude_spectrum, axis=0) / np.sum(magnitude_spectrum, axis=0)
        spectral_centroid = np.nanmean(spectral_centroid)
        
        spectral_bandwidth = np.sqrt(np.sum(((f[freq_mask, np.newaxis] - spectral_centroid) ** 2) * magnitude_spectrum, axis=0) / np.sum(magnitude_spectrum, axis=0))
        spectral_bandwidth = np.nanmean(spectral_bandwidth)
        
        geometric_mean = np.exp(np.mean(np.log(magnitude_spectrum + 1e-10), axis=0))
        arithmetic_mean = np.mean(magnitude_spectrum, axis=0)
        spectral_flatness = geometric_mean / arithmetic_mean
        spectral_flatness = np.nanmean(spectral_flatness)
        
        features = {
            'SpectralCentroid': spectral_centroid,
            'Bandwidth': spectral_bandwidth,
            'SpectralFlatness': spectral_flatness,
            'Length': (end -start),
            'DeltaFreq': (max_freq - min_freq)
        }

    except Exception as e:
        print(f"An error occurred: {e} in file {path}")

    return features


def get_outliers_df(data: pd.DataFrame, 
                    exclude_outliers_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Identifies and returns a DataFrame of outliers without plotting.
    
    Args:
        data (pd.DataFrame): The full dataset containing 'Specie_ID' and 'Distance'.
        exclude_outliers_df (pd.DataFrame, optional): A DataFrame of outliers to exclude. Default is None.
    
    Returns:
        pd.DataFrame: A DataFrame of the identified outliers.
    """
    outliers_list = []  # List to store outliers for DataFrame

    for label, group in data.groupby('Specie_ID'):
        Q1 = group['Distance'].quantile(0.25)
        Q3 = group['Distance'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR

        outliers = group[group['Distance'] > upper_bound]

        if exclude_outliers_df is not None:
            exclude_tuples = list(exclude_outliers_df[['Specie_ID', 'Distance']].itertuples(index=False, name=None))
            
            # Only apply the exclusion if exclude_tuples is not empty
            if exclude_tuples:
                outliers = outliers[~outliers.set_index(['Specie_ID', 'Distance']).index.isin(pd.MultiIndex.from_tuples(exclude_tuples))]

        # Append outliers to list if any were found
        if not outliers.empty:
            outliers_list.append(outliers)
    
    # Concatenate the list of outliers into a DataFrame (handle case when outliers_list is empty)
    if outliers_list:
        outliers_df = pd.concat(outliers_list)
        outliers_df = outliers_df.loc[:, ['Lonely Index', 'Distance', 'Specie_ID']]

        # Remove exclude_outliers_df from outliers_df
        if exclude_outliers_df is not None:
            exclude_index = exclude_outliers_df.set_index(['Lonely Index', 'Distance', 'Specie_ID']).index
            outliers_df = outliers_df[~outliers_df.set_index(['Lonely Index', 'Distance', 'Specie_ID']).index.isin(exclude_index)]

        return outliers_df
    else:
        # Return an empty DataFrame if no outliers were found
        return pd.DataFrame(columns=['Lonely Index', 'Distance', 'Specie_ID'])
    
    
def merge_distance_column(df_reduced, lonely_observations_df):
    """
    Merges a 'Distance' column into the dataset based on 'Lonely Index' from lonely observations.

    Parameters:
    - df_reduced (pd.DataFrame): Full dataset with a unique index for each observation.
    - lonely_observations_df (pd.DataFrame): DataFrame containing 'Lonely Index' and 'Distance' columns.

    Returns:
    - pd.DataFrame: Modified DataFrame with a 'Distance' column indicating the distance for lonely observations.
    """

    # Create a copy of the full dataset
    data_full = df_reduced.copy()

    # Initialize a new column 'Distance' and set it to 0 for all observations
    data_full['Distance'] = 0

    # Merge based on the index of data_full and the 'Lonely Index' from lonely_observations_df
    data_full = data_full.merge(lonely_observations_df[['Lonely Index', 'Distance']], 
                                left_on=data_full.index, right_on='Lonely Index', 
                                how='left', suffixes=('', '_lonely'))

    # If the 'Distance_lonely' column exists after the merge, use it to update the original 'Distance' column
    if 'Distance_lonely' in data_full.columns:
        data_full['Distance'] = data_full['Distance_lonely']
        data_full.drop(columns=['Distance_lonely'], inplace=True)

    # Ensure 'Distance' is numeric and fill missing values with 0
    data_full['Distance'] = pd.to_numeric(data_full['Distance'], errors='coerce').fillna(0)

    return data_full


def process_lonely_observations(df: pd.DataFrame, observations_df: pd.DataFrame):
    """
    Processes lonely observations by categorizing them into high, moderate, and low review groups.

    Parameters:
    - df (pd.DataFrame): Full dataset containing all observations.
    - observations_df (pd.DataFrame): DataFrame with lonely observations ('Lonely Index', 'Distance', 'Specie_ID').

    Returns:
    - tuple: Four DataFrames for each review category:
        - observations_high_review (pd.DataFrame)
        - observations_moderate_review (pd.DataFrame)
        - observations_moderate_high_review (pd.DataFrame)
        - observations_low_review (pd.DataFrame)
    """
    # Step 1: Merge distance column
    data_distance_df = merge_distance_column(df, observations_df)
    
    # Step 2: Calculate high review outliers from lonely_observations_df
    observations_high_review = get_outliers_df(observations_df)
    
    # Step 3: Calculate moderate review outliers (excluding high review outliers, using the full dataset)
    observations_moderate_review = get_outliers_df(data_distance_df, observations_high_review)
    
    # Step 4: Calculate moderate-high review outliers (without excluding any, using the full dataset)
    # Check if observations_high_review and observations_moderate_review are empty before concatenating

    if not observations_high_review.empty and not observations_moderate_review.empty:
        observations_moderate_high_review = pd.concat([observations_high_review, observations_moderate_review]).drop_duplicates()
    elif not observations_high_review.empty:
        # If only observations_high_review has data, use it as is
        observations_moderate_high_review = observations_high_review
    elif not observations_moderate_review.empty:
        # If only observations_moderate_review has data, use it as is
        observations_moderate_high_review = observations_moderate_review
    else:
        # If both are empty, return an empty DataFrame with the same columns
        observations_moderate_high_review = pd.DataFrame(columns=observations_high_review.columns)
        
    # Step 5: Calculate low review observations by excluding moderate-high review observations
    observations_low_review = observations_df.loc[:, ['Lonely Index', 'Distance', 'Specie_ID']]
    observations_low_review = observations_low_review[
        ~observations_low_review.set_index(['Lonely Index', 'Distance', 'Specie_ID']).index.isin(
            observations_moderate_high_review.set_index(['Lonely Index', 'Distance', 'Specie_ID']).index
        )
    ]

    # Return the three DataFrames
    return observations_high_review, observations_moderate_review, observations_moderate_high_review, observations_low_review


def modify_species_id(df, random_state):
    """
    Modifies the 'Specie_ID' column in a random sample (10%) of the DataFrame by replacing 
    the original values with different random values from the unique species.

    Parameters:
    - df (pd.DataFrame): DataFrame containing at least a 'Specie_ID' column with species labels.
    - random_state (int): Seed for reproducibility of the random sampling. Default is 42.

    Returns:
    - df_modified (pd.DataFrame): DataFrame with modified 'Specie_ID' values for the selected sample.
    - modifications (pd.DataFrame): DataFrame listing the index, original, and modified values of 'Specie_ID' in the sample.
    """
    
    # Set the random seed
    np.random.seed(random_state)
    
    # Get the list of unique species directly from the 'Specie_ID' column
    possible_species = df['Specie_ID'].unique()

    # Calculate the sample size (10% of the dataframe) & take a random sample of 10% of the dataframe
    sample_size = int(0.1 * len(df))
    sample_df = df.sample(n=sample_size, random_state=random_state)

    # Save the indices of the sample before modification
    modified_indices = sample_df.index.tolist()

    # Modify the 'Specie_ID' values in the sample with new random values
    def get_different_value(original_value, possible_values):
        new_value = original_value
        while new_value == original_value:
            new_value = np.random.choice(possible_values)
        return new_value

    sample_df['Specie_ID'] = sample_df['Specie_ID'].apply(lambda x: get_different_value(x, possible_species))

    # Replace the modified values back into a copy of the original dataframe
    df_modified = df.copy()
    df_modified.update(sample_df)

    # Create a DataFrame with the index, original value, and modified value
    modifications = pd.DataFrame({
        'Index': modified_indices,
        'Original': [df.loc[idx, 'Specie_ID'] for idx in modified_indices],
        'Modified': [df_modified.loc[idx, 'Specie_ID'] for idx in modified_indices]
    })

    # Return the modified dataframe and the DataFrame of modifications
    return df_modified, modifications
