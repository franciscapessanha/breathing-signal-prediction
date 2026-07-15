# Breathing Signal Prediction

Codebase for paper submitted to Interspeech2025 - "Breathing signal prediction from speech -  state-of-the-art and reproducibility"

![diagrams_interspeech2025_v2](https://github.com/user-attachments/assets/87dd4864-cfa6-47e7-be8e-012861386dad)


## Installation

1. **Basic Installation**: `pip install -r requirements.txt`
2. **Virtual Environment Setup (Recommended)**:
   - Uses `venv` to create an isolated environment.
   - Provides activation steps for both macOS/Linux and Windows.
   
## Structure
src
 |- evaluate_on_test.py      # Evaluates the proposed model on the test set
 |- data_preparation.py      # Functions necessary for data preparation
 |- model_definition.py      # Model implementation
 |- __init__.py
 |- utils.py                 # Various functions to assist with cross-validation and prediction
 |- train_and_eval.py        # Trains and evaluates models in a cross-validation setting
 |- config.py                # Defines common paths and experimental parameters for the models reported in our paper
 |- cross_validation_train.py # Functions related to cross-validation and evaluation

data
 |- wav                      # Folder containing audio (.wav) files
 |- ground_truth             # Folder containing "labels.csv," which holds breathing signal values for all samples
 |- models                   # Folder where models will be saved
 |- output                   # Folder where predicted breathing signals will be saved

## Trained models

*insert link to drive*
# breathing-signal-prediction
