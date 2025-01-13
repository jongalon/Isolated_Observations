# Project description

Welcome to the Isolated Observations project! This repository provides tools and methods to help researchers and bioacoustics experts efficiently identify and review audio segments in passive acoustic monitoring (PAM) datasets that may have been mislabeled or misclassified. By improving the quality of labels, we can enhance the performance of species detection and classification models in bioacoustic studies.

## Why This Project?

Labeling audio segments in large datasets is a time-consuming process, often prone to errors. This project aims to streamline the review process by flagging potentially problematic segments for human verification. Instead of reviewing entire datasets, experts can focus on segments that are more likely to require correction.

The project's methods use feature extraction, latent variable identification, and hierarchical clustering to isolate audio segments that deviate from others with the same label. This helps reduce false automated labeling and improves the overall quality of PAM datasets.

## Project Structure

The repository is organized as follows:

- **notebooks/**: Contains Jupyter notebookswith step-by-step implementations of the main processes.
  - `1_ExtractFeatures.ipynb`:  Extracts features such as cepstral coefficients, spectral features, and time-frequency characteristics from selection tables (in Excel format). Originally written in MATLAB, this feature extraction was ported to Python for broader accessibility. The output is a matrix combining selection table data with extracted features.
  - `2_Find_Latent_Variables.ipynb`: Performs dimensionality reduction on the feature matrix from notebook 1, identifying latent variables that help explain the dataset's structure. This process adapts to each dataset, ensuring relevant features are emphasized. The output is a reduced feature matrix.
  - `3_FindIsolatedObservations.ipynb`: Detects isolated observations based on the reduced feature matrix. The output is an Excel file listing the flagged observations, including their original indices to simplify the review process.
  - `Create_Selection_Table_from_Isolated.ipynb`: Generates selection tables in RAVEN format, highlighting potentially problematic segments for easier review within the RAVEN software.

- **src/**: Contains modularized code for processing data, performing clustering, and visualizing results.
- **data/**: Stores input data, including selection tables and feature matrices for different datasets.
- **audios/**: Includes audio files used in the analysis. Dataset 1 corresponds to subsets A and C, and Dataset 2 to subset B.
- **output/**: Stores results:
  - The reduced matrix of latent variables after dimensionality reduction.
  - Lists of isolated observations (`.xlsx` format).
  - Flagged observations in labeled data.
  - Selection tables for individual and combined review sessions in RAVEN.

## Installation

Follow these steps to set up the project:

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/jongalon/Isolated_Observations
   cd Isolated_Observations
   ```

2. Install the required dependencies:

   Make sure you have [Python 3.11](https://www.python.org/downloads/) installed. Then, use the following command to install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. MATLAB setup:
   If you prefer to use MATLAB for feature extraction, ensure MATLAB is installed. You can run the `1_ExtractFeatures_fromSelectionTables.m` script for this purpose.

## Usage

Once the setup is complete, you can start analyzing your audio data by following these steps:

1. Place your selection tables files in the `data/` folder.
2. Place our audio files in the `audios/` folder.
3. Run the notebooks in the following order::
   - `1_ExtractFeatures.ipynb` or
      `1_ExtractFeatures_fromSelectionTables.m` (MATLAB)
   - `2_Find_Latent_Variables.ipynb`
   - `3_FindIsolatedObservations.ipynb`
   - `Create_Selection_Table_from_Isolated.ipynb`
4. Review the flagged observations using the output in the output/ folder or directly within the RAVEN software.

## Contributions

If you would like to contribute, please follow these steps:

1. Open an issue to discuss the changes or improvements you intend to make.
2. Submit a pull request with your proposed changes.

Contributions are always welcome, and we appreciate any help in improving the project!

