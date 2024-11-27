# Isolated Observations

This project provides methods and tools for analyzing and detecting isolated labeled observations of vocalizations in bioacoustic audio data. It focuses on identifying latent variables and performing hierarchical clustering to better understand the structure of the data.

## Project Structure

- **notebooks/**: Contains Jupyter notebooks used in the project.
  - `Find_Latent_Variables_NoCommunalities.ipynb`: Identifies latent variables without considering communalities.
  - `FindIsolatedObservations.ipynb`: Detects isolated observations in the dataset.
  - `Create_Selection_Table_from_Isolated.ipynb`: Creates a selection table for reviewing isolated observations.
- **src/**: Contains the modularized project code with data processing, modeling, and visualization functions.
- **data/**: Directory for storing data files. It contains selection tables from RAVEN for three subsets with the calculated features. One of the subsets was obtained from an automatically labeled dataset using KOOGU.
- **tests/**: Contains unit tests to verify code functionality and ensure reliability.
- **audios/**: Contains the audio files used for this project. Dataset 1 is used for subsets A and C, while Dataset 2 is used for subset B.
- **output/**: Contains the results per Subset, including:
  - The list of isolated observations in `.xlsx` format.
  - All labeled data with flagged observations for review.
  - Selection tables for each audio, including a column for review.
  - A combined selection table for use within RAVEN, allowing all audios to be reviewed in a single session.

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

## Usage

- Place the data files in the `data/` folder.
- Run the notebooks in the `notebooks/` folder to explore and analyze the data. It is recommended to follow this order:
  1. `Find_Latent_Variables_NoCommunalities.ipynb`
  2. `FindIsolatedObservations.ipynb`
  3. `Create_Selection_Table_from_Isolated.ipynb`
- Use the functions in `src/` for specific analyses or to integrate them into other projects.

## Contributions

If you would like to contribute, please follow these steps:

1. Open an issue to discuss the changes or improvements you intend to make.
2. Submit a pull request with your proposed changes.

Contributions are always welcome, and we appreciate any help in improving the project!

## Author

Jonathan Gallego Londoño  
PhD Student in Electronic and Computer Engineering  
Universidad de Antioquia
