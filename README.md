# Isolated Observations

This project provides methods and tools for efficiently detecting potentially mislabeled and misclassified audio segments in PAM bioacoustic datasets. It enables experts to review flagged segments and take corrective actions, improving the overall quality of species vocalization detection and classification models.

By focusing on feature extraction, latent variable identification, and hierarchical clustering, the project isolates segments that deviate from others with the same label. This refinement supports human oversight by reducing false automated labeling, making targeted reviews more efficient.

## Project Structure

- **notebooks/**: Contains Jupyter notebooks and Matlab scripts used in the project.
  - `1_ExtractFeatures_fromSelectionTables.m`:  Extracts linear cepstral coefficients using MATLAB.
  - `2_Find_Latent_Variables.ipynb`: Performs dimensionality reduction and identifies latent variables.
  - `3_FindIsolatedObservations.ipynb`:Detects isolated observations and evaluates model performance.
  - `Create_Selection_Table_from_Isolated.ipynb`: Creates selection tables in RAVEN format for reviewing isolated observations.
- **src/**: Modularized code for data processing, modeling, and visualization.
- **data/**: Contains input data, including selection tables and calculated features for subsets derived from datasets.
- **audios/**: Includes audio files used in the analysis. Dataset 1 corresponds to subsets A and C, and Dataset 2 to subset B.
- **output/**: Stores results:
  - Lists of isolated observations (`.xlsx` format).
  - Flagged observations in labeled data.
  - Selection tables for individual and combined review sessions in RAVEN.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/jongalon/Isolated_Observations
   cd Isolated_Observations
   ```

2. Install the dependencies:

   Make sure you have [Python 3.11](https://www.python.org/downloads/) installed. Then, use the following command to install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. MATLAB setup:
   Make sure MATLAB is installed to run the script `1_ExtractFeatures_fromSelectionTables.m`. 

## Usage

1. Place your data files in the `data/` folder.
2. Follow these steps for analysis:
   - Run `1_ExtractFeatures_fromSelectionTables.m` in MATLAB.
   - Execute the Jupyter notebooks in this order:
     1. `2_Find_Latent_Variables.ipynb`
     2. `3_FindIsolatedObservations.ipynb`
     3. `Create_Selection_Table_from_Isolated.ipynb`
3. Review flagged observations using the output in the `output/` folder or directly in RAVEN.

## Contributions

If you would like to contribute, please follow these steps:

1. Open an issue to discuss the changes or improvements you intend to make.
2. Submit a pull request with your proposed changes.

Contributions are always welcome, and we appreciate any help in improving the project!

## Author
