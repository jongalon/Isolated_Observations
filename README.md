
# Isolated Observations

This project provides methods and tools for analyzing and detecting isolated labeled observations of vocalizations in bioacoustic audio data. It focuses on identifying latent variables and performing hierarchical clustering.

## Project Structure

- **notebooks/**: Contains Jupyter notebooks used in the project.
  - `Find_Latent_Variables.ipynb`
  - `FindIsolatedObservations.ipynb`
- **src/**: Contains the modularized project code with data processing, modeling, and visualization functions.
- **data/**: Directory for storing data file. It has selection tables from RAVEN of 2 datasets and from an automatic labeled dataset from KYOOGU.
- **tests/**: Contains short unit tests to verify code functionality.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/jongalon/Isolated_Observations
   cd your_project_name
   ```

2. Install the dependencies:

   Make sure you have [Python 3.11](https://www.python.org/downloads/) installed. Then, use the following command to install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

- Place the data files in the `data/` folder.
- Run the notebooks in the `notebooks/` folder to explore and analyze the data.
- Use the functions in `src/` for specific analyses or to integrate them into other projects.

## Contributions

If you would like to contribute, please open an "Issue" or submit a "Pull Request."

## Author

Jonathan Gallego Londoño  
PhD student in Electronic and Computer Engineering  
Universidad de Antioquia
