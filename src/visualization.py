# src/visualization.py
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.lines as mlines
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
from matplotlib.lines import Line2D
import networkx as nx

def plot_factor_loadings(loadings_df, factor_size=4000, variable_size=3500, font_size=14):
    """
    Plots a graph representation of factor loadings for a factor analysis.

    Parameters:
        loadings_df (DataFrame): A DataFrame containing the factor loadings, with variables as rows and factors as columns.
        factor_size (int, optional): Size of the factor nodes in the plot. Default is 4000.
        variable_size (int, optional): Size of the variable nodes in the plot. Default is 3500.
        font_size (int, optional): Font size for labels in the plot. Default is 14.

    Returns:
        None
    """
    # Create the graph
    G = nx.DiGraph()

    # Define the spacing between nodes
    space = 1  # Increase the spacing between variables of the same factor
    group_space_multiplier = 0.1  # Further reduce the extra space between different factor groups

    # Create a custom layout
    pos = {}
    y_offset_factors = 0  # Initial y offset for factors
    y_offset_variables = 0  # Initial y offset for variables

    # Assign each variable to the factor with the highest loading
    assignment = {}
    for variable in loadings_df.index:
        max_loading_factor = loadings_df.loc[variable].abs().idxmax()
        max_loading_value = loadings_df.loc[variable, max_loading_factor]
        if abs(max_loading_value) > 0.35:
            assignment[variable] = (max_loading_factor, max_loading_value)

    # Create factor nodes and assign initial positions
    factor_positions = {}
    factor_spacing = len(assignment) * space / len(loadings_df.columns)  # Adjust spacing
    for factor_index, factor in enumerate(loadings_df.columns):
        factor_name = f'Factor {factor_index + 1}'
        G.add_node(factor_name, color='red', size=factor_size)
        factor_positions[factor_name] = y_offset_factors
        pos[factor_name] = (1, y_offset_factors)
        y_offset_factors += factor_spacing

    # Position and add variables to the graph, grouping by factor
    for factor_index, factor in enumerate(loadings_df.columns):
        factor_name = f'Factor {factor_index + 1}'

        # Assign positions to variables assigned to this factor
        for variable, (assigned_factor, weight) in assignment.items():
            if assigned_factor == factor:
                pos[variable] = (0, y_offset_variables)
                G.add_node(variable, color='lightblue', size=variable_size)
                G.add_edge(variable, factor_name, weight=round(weight, 1))
                y_offset_variables += space

        # Add extra space between groups
        y_offset_variables += space * group_space_multiplier

    # Colors and sizes
    node_colors = [nx.get_node_attributes(G, 'color').get(node, 'lightblue') for node in G.nodes()]
    node_sizes = [nx.get_node_attributes(G, 'size').get(node, variable_size) for node in G.nodes()]
    edge_labels = nx.get_edge_attributes(G, 'weight')

    # Draw the graph
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=node_sizes, edge_color='black', width=2, font_size=font_size, font_color='black', font_weight='bold')

    # Edge labels with one decimal place
    for (u, v), weight in edge_labels.items():
        x_pos = (pos[u][0] + pos[v][0]) / 2 + 0.02  # Horizontal adjustment
        y_pos = (pos[u][1] + pos[v][1]) / 2
        plt.text(x_pos, y_pos, f'{weight:.1f}', color='red', fontsize=font_size, fontweight='bold')

    plt.title('Factor Analysis Loadings')
    plt.show()



def plot_dendrogram_lonelylabels(data, only_features_scaled, lonely_observations):
    """
    Plots a dendrogram for the dataset, highlighting lonely observations with red labels.

    Parameters:
    - data (pd.DataFrame): DataFrame containing at least 'Specie_ID' and an index representing observations.
    - only_features_scaled (pd.DataFrame): Scaled features corresponding to 'data' observations.
    - lonely_observations (list of tuples): List of (index, specie) pairs for lonely observations.
    - color_threshold (float): Threshold to color branches in the dendrogram. Default is 13.

    Returns:
    None. Displays a dendrogram plot highlighting lonely observations in red.
    """

    # Create labels for the observations, highlighting the lonely ones
    color_threshold = 13 #This number is selected by the user.

    labels = data['Specie_ID'].tolist()
    data_features = only_features_scaled.loc[data.index]

    # Calculate the dendrogram for the observations in the entire dataset
    Z_data = linkage(data_features, method='ward')
   
    # Get the order of the observations in the dendrogram
    dendro = dendrogram(Z_data, labels=data.index, no_plot=True)
    leaves_order = dendro['leaves']
    
    # Sort the observations according to the order of the leaves in the dendrogram
    data_sorted = data.iloc[leaves_order]

    labels_indices = [
        f'**{data.index[i]}**' if data.index[i] in [idx for idx, _ in lonely_observations] else str(data.index[i]) 
        for i in range(len(data))
    ]

    labels_names = [
        f"**{data.iloc[i]['Specie_ID']}**" if data.index[i] in [idx for idx, _ in lonely_observations] else str(data.iloc[i]['Specie_ID'])
        for i in range(len(data_sorted))
    ]

    # Create combined labels for each observation in the current cluster
    combined_labels = [
        f"{data.iloc[i]['Specie_ID']} ({data.index[i]})"
        for i in range(len(data))
    ]

    fig, ax= plt.subplots(figsize=(5, .15 * len(data_sorted))) 
    #dendrogram(Z_data, labels=labels_indices, color_threshold=color_threshold, orientation='right',  ax=axs[0], leaf_font_size=7)
    dendro = dendrogram(Z_data, labels=combined_labels, color_threshold=color_threshold, orientation='right', ax=ax, leaf_font_size=7)
    
    # Set the title and axis labels
    ax.set_title(f'Dendrogram')
    ax.set_ylabel('Label (Index)')
    ax.set_xlabel('Distance')
    ax.set_xlim([0, color_threshold])
    
    # Change the color of the lonely labels to red
    for label in ax.get_yticklabels():
        text = label.get_text()
        index_in_text = int(text.split('(')[-1][:-1])  # Extract the index from the combined label
        if index_in_text in [idx for idx, _ in lonely_observations]:  # Check if it's a lonely observation
            label.set_color('red')
    

    # Adjust the space between subplots
    plt.tight_layout()
    plt.show()

# General function to collect outlier positions
def collect_outlier_positions(data, outliers):
    """
    Collects the x, y positions and index of outliers from the data.
    
    Args:
        data (pd.DataFrame): The full dataset containing 'Specie_ID', 'Distance', and 'Lonely Index'.
        outliers (pd.DataFrame): The subset of data identified as outliers.
    
    Returns:
        list: A list of tuples containing x (Specie_ID), y (Distance), and the lonely index of each outlier.
    """
    outlier_positions = []
    for i in outliers.index:
        x_value = data.loc[i, 'Specie_ID']
        y_value = data.loc[i, 'Distance']
        lonely_index_value = data.loc[i, 'Lonely Index']
        outlier_positions.append((x_value, y_value, lonely_index_value))
    return outlier_positions


# General function to annotate outliers
def annotate_outliers(outlier_positions, circle_positions, show_annotations=True, offset=1.5):
    """
    Annotates the outliers on the plot by adding red labels, avoiding overlap.
    
    Args:
        outlier_positions (list): List of tuples with outlier positions (x, y, index).
        circle_positions (list): List to track already placed annotations to avoid overlap.
        show_annotations (bool): Whether to display the annotations or not. Default is True.
        offset (int): The vertical offset to adjust label positioning to avoid overlaps. Default is 5.
    
    Returns:
        None
    """
    if not show_annotations:
        return  # If annotations are not to be shown, skip this step
    
    for (x_value, y_value, lonely_index_value) in outlier_positions:
        new_y_position = y_value
        for circle_x, circle_y in circle_positions:
            if circle_x == x_value and abs(new_y_position - circle_y) < 35:
                if new_y_position >= circle_y:
                    new_y_position = y_value + offset
                else:
                    new_y_position = y_value - offset
        plt.text(x=x_value, y=new_y_position, s=lonely_index_value, 
                 color='red', fontsize=9, ha='center', 
                 va='bottom' if new_y_position >= y_value else 'top', fontweight='bold')
        circle_positions.append((x_value, new_y_position))


# General function to create the boxplot, scatter plot for outliers, and return a DataFrame of outliers
def plot_boxplot_with_outliers(data: pd.DataFrame, 
                               title: str, 
                               outlier_color: str, 
                               legend_label: str, 
                               boxplot_legend_text: str, 
                               show_annotations: bool = True, 
                               exclude_outliers_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Creates a boxplot with custom outlier annotations and returns a DataFrame of identified outliers.
    
    Args:
        data (pd.DataFrame): The full dataset containing 'Specie_ID' and 'Distance'.
        title (str): The title of the plot.
        outlier_color (str): The color of the outlier markers.
        legend_label (str): The label for the outlier legend entry.
        boxplot_legend_text (str): The label for the boxplot legend entry.
        show_annotations (bool): Whether to show annotations for the outliers. Default is True.
        exclude_outliers_df (pd.DataFrame, optional): A DataFrame of outliers to exclude from the plot. Default is None.
    
    Returns:
        pd.DataFrame: A DataFrame of the identified outliers.
    """
    flierprops = dict(marker='None')

    # Create the boxplot (no outliers shown by default)
    plt.figure(figsize=(10, 6))
    boxplot = sns.boxplot(x='Specie_ID', y='Distance', data=data, whis=1.5, flierprops=flierprops, order=sorted(data['Specie_ID'].unique()))

    # Set labels and title
    plt.xlabel('Label')
    plt.ylabel('Distance to Nearest Same Label')
    plt.title(title)

    # Rotate x-axis labels if necessary and set font style to italic
    plt.xticks(rotation=90, fontstyle='italic')

    # Track circle positions and collect all outlier positions
    circle_positions = []
    all_outlier_positions = []
    outliers_list = []  # List to store outliers for DataFrame

    # Identify and plot outliers for each group
    for label, group in data.groupby('Specie_ID'):
        Q1 = group['Distance'].quantile(0.25)
        Q3 = group['Distance'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR

        outliers = group[group['Distance'] > upper_bound]
        
        if exclude_outliers_df is not None:
            exclude_tuples = list(exclude_outliers_df[['Specie_ID', 'Distance']].itertuples(index=False, name=None))
            outliers = outliers[~outliers.set_index(['Specie_ID', 'Distance']).index.isin(pd.MultiIndex.from_tuples(exclude_tuples))]

        # Plot outliers with the specified color
        plt.scatter(outliers['Specie_ID'], outliers['Distance'], 
                    color=outlier_color, edgecolor='black', s=45, zorder=3)

        # Collect all outlier positions for this group
        outlier_positions = collect_outlier_positions(data, outliers)
        all_outlier_positions.extend(outlier_positions)

        # Append outliers to list
        outliers_list.append(outliers)

        # Track original circle positions
        for i in outliers.index:
            circle_positions.append((data.loc[i, 'Specie_ID'], data.loc[i, 'Distance']))

    # Concatenate the list of outliers into a DataFrame
    outliers_df = pd.concat(outliers_list)
    outliers_df = outliers_df.loc[:, ['Lonely Index', 'Distance', 'Specie_ID']]

    # Remove exclude_outliers_df from outliers_df, circle_positions, and all_outlier_positions
    if exclude_outliers_df is not None:
        exclude_index = exclude_outliers_df.set_index(['Lonely Index', 'Distance', 'Specie_ID']).index

        # Remove from outliers_df
        outliers_df = outliers_df[~outliers_df.set_index(['Lonely Index', 'Distance', 'Specie_ID']).index.isin(exclude_index)]

        # Convert exclude_outliers_df to tuples for comparison
        exclude_tuples = list(exclude_outliers_df[['Specie_ID', 'Distance']].itertuples(index=False, name=None))

        # Remove from circle_positions (tuples of (Specie_ID, Distance))
        circle_positions = [pos for pos in circle_positions if pos not in exclude_tuples]

        # Remove from all_outlier_positions (tuples of (Specie_ID, Distance, Lonely Index))
        all_outlier_positions = [pos for pos in all_outlier_positions if (pos[0], pos[1]) not in exclude_tuples]

    # Annotate the outliers (optional)
    annotate_outliers(all_outlier_positions, circle_positions, show_annotations)

    # Create the legend
    boxplot_color = sns.color_palette()[0]
    outlier_legend = mlines.Line2D([], [], color=outlier_color, marker='o', markersize=10, linestyle='None', markeredgecolor='black', label=legend_label)
    boxplot_legend = mlines.Line2D([], [], color=boxplot_color, marker='s', linestyle='None', label=boxplot_legend_text)

    # Prepare the list of legend handles
    legend_handles = [outlier_legend, boxplot_legend]

    # Conditionally add the red text line for index if annotations are shown
    if show_annotations:
        red_text_line = mlines.Line2D([], [], color='red', marker='*', linestyle='None', label='Index')
        legend_handles.append(red_text_line)

    # Add the legend to the plot
    plt.legend(handles=legend_handles)

    # Show the plot
    plt.show()

    # Return the DataFrame of outliers
    return outliers_df

def plot_confusion_matrix(cm, title='Confusion Matrix for Lonely Observations', labels=['Not Modified', 'Modified'], cmap='Blues'):
    """
    Plots the confusion matrix with custom formatting.

    Parameters:
    - cm: array-like of shape (n_classes, n_classes)
        Confusion matrix to be displayed.
    - title: str, default='Confusion Matrix for Lonely Observations'
        Title for the plot.
    - labels: list, default=['Not Modified', 'Modified']
        Labels for the classes.
    - cmap: str, default='Blues'
        Colormap for the plot.
    """
    fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=cmap, ax=axes)
    plt.title(title, fontsize=16)
    axes.set_xlabel('Predicted label', fontsize=14)
    axes.set_ylabel('True label', fontsize=14)
    
    # Adjust tick labels
    for tick in axes.get_xticklabels():
        tick.set_fontsize(10)
        tick.set_rotation(45)  # Rotate the x-axis labels
    
    axes.tick_params(axis='y', labelsize=10)

    # Increase the font size of the numbers in the matrix
    for text in disp.text_.ravel():
        text.set_fontsize(12)  
    plt.show()


    import pandas as pd
import matplotlib.pyplot as plt

# def plot_outliers(df_reduced, lonely_observations_df):
#     # Merge the distance column
#     # Create a copy of the full dataset
#     df_reduced = df_reduced.copy()

#     # Initialize a new column 'Distance' and set it to 0 for all observations
#     df_reduced['Distance'] = 0

#     # Merge based on the index of data_full and the 'Lonely Index' from lonely_observations_df
#     df_reduced = df_reduced.merge(lonely_observations_df[['Lonely Index', 'Distance']], 
#                                 left_on=df_reduced.index, right_on='Lonely Index', 
#                                 how='left', suffixes=('', '_lonely'))

#     # If the 'Distance_lonely' column exists after the merge, use it to update the original 'Distance' column
#     if 'Distance_lonely' in df_reduced.columns:
#         df_reduced['Distance'] = df_reduced['Distance_lonely']
#         df_reduced.drop(columns=['Distance_lonely'], inplace=True)

#     # Ensure 'Distance' is numeric and fill missing values with 0
#     df_reduced['Distance'] = pd.to_numeric(df_reduced['Distance'], errors='coerce').fillna(0)


#     #df_reduced_withdistance = merge_distance_column(df_reduced, lonely_observations_df)
    
#     # Calculate quartiles and interquartile range (IQR)
#     Q1 = df_reduced['Distance'].quantile(0.25)
#     Q3 = df_reduced['Distance'].quantile(0.75)
#     IQR = Q3 - Q1

#     # Define the upper bound for outliers
#     upper_bound = Q3 + 1.5 * IQR

#     # Identify the outliers
#     outliers = df_reduced[df_reduced['Distance'] > upper_bound]
    
#     # Select the top 3 outliers with the highest distance
#     top_outliers = outliers.nlargest(3, 'Distance')

#     # Outlier properties in the boxplot
#     flierprops = dict(marker='o', markerfacecolor='#aed768', markersize=8, linestyle='none')

#     # Create the boxplot
#     plt.figure(figsize=(2, 7))
#     plt.boxplot(df_reduced['Distance'], vert=True, flierprops=flierprops, positions=[1])
#     plt.title('')
#     plt.gca().set_xticks([1])
#     plt.gca().set_xticklabels(['Isolated Observations'])
#     plt.xticks(rotation=90, fontsize=12)
#     plt.ylabel('Distance', fontsize=12)


#     # Add labels to the top 3 outliers with the highest distance
#     for _, row in top_outliers.iterrows():
#         #plt.text(1.02, row['Distance'], f" {row['Lonely Index']}:{row['Specie_ID']}", 
#         #         verticalalignment='center', fontsize=9, color='black')
#         plt.text(1.02, row['Distance'], f" {row['Lonely Index']}", 
#                  verticalalignment='center', fontsize=9, color='black')

#     plt.show()

#     return outliers

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

def plot_outliers(df_reduced, lonely_observations_df, top_n=2):
    """
    Plots two subplots: a boxplot of distances showing outliers and a scatter plot
    of isolated observations by species, highlighting the top N isolated observations.

    Parameters:
    - df_reduced (DataFrame): Original DataFrame containing all species.
    - lonely_observations_df (DataFrame): DataFrame containing lonely observations with columns 'Lonely Index' and 'Distance'.
    - top_n (int): Number of top isolated observations to label per species in the right plot. Default is 2.
    """
    # Copy df_reduced and initialize 'Distance' column to zero
    df_reduced = df_reduced.copy()
    df_reduced['Distance'] = 0

    # Merge 'Distance' from lonely_observations_df to df_reduced based on index
    df_reduced = df_reduced.merge(lonely_observations_df[['Lonely Index', 'Distance']], 
                                  left_on=df_reduced.index, right_on='Lonely Index', 
                                  how='left', suffixes=('', '_lonely'))

    # Update 'Distance' column if 'Distance_lonely' exists
    if 'Distance_lonely' in df_reduced.columns:
        df_reduced['Distance'] = df_reduced['Distance_lonely']
        df_reduced.drop(columns=['Distance_lonely'], inplace=True)

    # Ensure 'Distance' is numeric and fill missing values with 0
    df_reduced['Distance'] = pd.to_numeric(df_reduced['Distance'], errors='coerce').fillna(0)

    # Calculate quartiles, IQR, and upper bound for outliers
    Q1 = df_reduced['Distance'].quantile(0.25)
    Q3 = df_reduced['Distance'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR

    # Identify and select the top 3 outliers by distance
    outliers = df_reduced[df_reduced['Distance'] > upper_bound]
    top_outliers = outliers.nlargest(3, 'Distance')

    # Set up the figure with subplots
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 14])

    # Left subplot - Boxplot with outliers
    ax1 = fig.add_subplot(gs[0])
    flierprops = dict(marker='o', markerfacecolor='#aed768', markersize=8, linestyle='none')
    ax1.boxplot(df_reduced['Distance'], vert=True, flierprops=flierprops, positions=[1])
    ax1.set_xticks([1])
    ax1.set_xticklabels(['Isolated Observations'], rotation=90, fontsize=12)
    ax1.set_ylabel('Distance', fontsize=12)
    ax1.set_ylim(-5, df_reduced['Distance'].max() * 1.1)  # Adjust y-axis for visibility

    # Add labels for top 3 outliers in the left plot
    for _, row in top_outliers.iterrows():
        ax1.text(1.02, row['Distance'], f" {row['Lonely Index']}", verticalalignment='center',
                 fontsize=9, color='black')

    # Right subplot - Scatter plot of isolated observations by species
    ax2 = fig.add_subplot(gs[1])

    # Ensure species are ordered by total count in df_reduced
    species_order = df_reduced['Specie_ID'].value_counts().index

    # Plot each outlier as a point in the sorted species order
    for specie_id in species_order:
        specie_outliers = outliers[outliers['Specie_ID'] == specie_id]
        
        # Check if there are isolated observations for this species
        if len(specie_outliers) > 0:
            # Plot all outliers for the species
            ax2.scatter([specie_id] * len(specie_outliers), specie_outliers['Distance'], 
                        alpha=0.55, edgecolor='black', s=75)
            
            # Get the top N outliers for this species based on Distance
            top_outliers = specie_outliers.nlargest(top_n, 'Distance')
            
            # Add labels for the top N outliers with alternate offsets to avoid overlap
            for j, (_, row) in enumerate(top_outliers.iterrows()):
                # Alternate y_offset for each point to avoid overlap
                y_offset = 4 if j % 2 == 0 else -4  # Adjust the offset as needed
                ax2.annotate(f"{row['Lonely Index']}", 
                             (specie_id, row['Distance']), 
                             textcoords="offset points", xytext=(16, y_offset), ha='center',
                             fontsize=10, color='red')
        else:
            # Plot a point with zero distance for species with no isolated observations
            ax2.scatter(specie_id, 0, alpha=0.0)  # Transparent point as placeholder

    # Custom legend elements for the right plot
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Isolated observations', 
               markerfacecolor='black', markersize=8, alpha=0.55),
        Line2D([0], [0], marker='*', color='w', label=f'Index of top {top_n} isolated observations', 
               markerfacecolor='red', markersize=10)
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=12)

    ax2.set_xlabel(' ', fontsize=14)
    ax2.set_ylabel(' ')
    ax2.set_ylim(-5, df_reduced['Distance'].max() * 1.1)
    ax2.set_xticks(range(len(species_order)))
    ax2.set_xticklabels(species_order, rotation=90, fontstyle='italic', fontsize=12)
    
    plt.tight_layout()
    plt.show()

    return outliers



def plot_isolated_observations_by_species(df_reduced, outliers, top_n=2):
    """
    Plots isolated observations by species, highlighting the top N isolated observations.

    Parameters:
    - df_reduced (DataFrame): Original DataFrame containing all species.
    - outliers (DataFrame): DataFrame containing outliers with columns 'Specie_ID', 'Distance', and 'Lonely Index'.
    - top_n (int): Number of top isolated observations to label per species. Default is 2.
    """
    # Ensure 'Distance' column is numeric
    outliers['Distance'] = pd.to_numeric(outliers['Distance'], errors='coerce')
    
    # Sort species by the total number of observations in df_reduced
    species_order = df_reduced['Specie_ID'].value_counts().index

    plt.figure(figsize=(12, 8))

    # Plot each outlier as a point in the sorted species order
    for specie_id in species_order:
        specie_outliers = outliers[outliers['Specie_ID'] == specie_id]
        
        # Check if there are isolated observations for this species
        if len(specie_outliers) > 0:
            # Plot all outliers for the species
            plt.scatter([specie_id] * len(specie_outliers), specie_outliers['Distance'], 
                        alpha=0.55, edgecolor='black', s=75)
            
            # Get the top N outliers for this species based on Distance
            top_outliers = specie_outliers.nlargest(top_n, 'Distance')
            
            # Add labels for the top N outliers with alternate offsets to avoid overlap
            for j, (_, row) in enumerate(top_outliers.iterrows()):
                # Alternate y_offset for each point to avoid overlap
                y_offset = 4 if j % 2 == 0 else -4  # Adjust the offset as needed
                plt.annotate(f"{row['Lonely Index']}", 
                             (specie_id, row['Distance']), 
                             textcoords="offset points", xytext=(16, y_offset), ha='center',
                             fontsize=10, color='red')
        else:
            # Plot a point with zero distance for species with no isolated observations
            plt.scatter(specie_id, 0, alpha=0.0)  # Transparent point as placeholder

    # Custom legend elements
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Isolated observations', 
               markerfacecolor='black', markersize=8, alpha=0.55),
        Line2D([0], [0], marker='*', color='w', label=f'Index of top {top_n} isolated observations', 
               markerfacecolor='red', markersize=10)
    ]

    # Add the custom legend inside the plot (upper right)
    plt.legend(handles=legend_elements, loc='upper right', fontsize=14)

    #plt.title('Isolated Observations by Species', fontsize=16)
    plt.xlabel('Species', fontsize=14)
    plt.ylabel('Distance', fontsize=14)
    plt.xticks(rotation=90, fontstyle='italic', fontsize=12)
    plt.show()
    
    
    # Example usage
    # plot_isolated_observations_by_species(outliers)


def plot_isolated_observations(df_reduced, lonely_observations_df):
    
    #Calculate the percentage of "lonely" observations
    num_lonely_observations = len(lonely_observations_df)
    num_total_observations = len(df_reduced)
    percentage_lonely = (num_lonely_observations / num_total_observations) * 100
    
    # Calculate species order by average distance
    #species_order = lonely_observations_df.groupby('Specie_ID')['Distance'].mean().sort_values(ascending=False).index
    species_order = df_reduced['Specie_ID'].value_counts().index

    # Reorder total counts and isolated counts based on species_order
    total_counts_df_reduced = df_reduced['Specie_ID'].value_counts().reindex(species_order, fill_value=0)
    isolated_counts_lonely = lonely_observations_df['Specie_ID'].value_counts().reindex(species_order, fill_value=0)
    # Calculate the percentage of isolated observations for each Specie_ID
    percentage_isolated = (isolated_counts_lonely / total_counts_df_reduced * 100).reindex(species_order, fill_value=0)

    # Create an index array to align the bars
    indices = np.arange(len(total_counts_df_reduced))
    bar_width = 0.65

    # Set up the figure with adjusted subplot widths
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 14])

    # Left plot (smaller)
    ax1 = fig.add_subplot(gs[0])
    label = 'Original observations'
    
    num_total_observations = total_counts_df_reduced.sum()  # Example total for demo
    ax1.bar(label, 100, color='#d3d3d3')
    ax1.bar(label, percentage_lonely, color='#5ce1e6')
    ax1.text(0, 100, f'{num_total_observations}', ha='center', va='bottom', fontsize=13, color='gray', fontweight='bold')
    ax1.text(0, percentage_lonely - 5, f'{percentage_lonely:.1f}%', ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')
    ax1.set_ylabel('Percentage (%)', fontsize=14)
    ax1.set_xticks([0])
    ax1.set_xticklabels([label], rotation=90, fontsize=12)
    ax1.set_ylim(0, 130)

    # Right plot (larger)
    ax2 = fig.add_subplot(gs[1])
    ax2.bar(indices, [100] * len(total_counts_df_reduced), width=bar_width, color='#d3d3d3', label='Original observations')
    ax2.bar(indices, percentage_isolated, width=bar_width, color='#5ce1e6', label='Percentage of isolated observations')
    for idx, (specie, count) in enumerate(total_counts_df_reduced.items()):
        ax2.text(idx + 0.25, 100, f'{count}', ha='center', va='bottom', fontsize=13, color='gray', fontweight='bold')
    for idx, percentage in enumerate(percentage_isolated):
        position = percentage if percentage < 3 else percentage - 4.5
        ax2.text(idx, position, f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')
    ax2.set_xticks(indices)
    ax2.set_xticklabels(species_order, rotation=90, fontsize=12, fontstyle='italic')
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 130)

    plt.tight_layout()
    plt.show()

# Example usage
#plot_isolated_observations(df_reduced, lonely_observations_df)
