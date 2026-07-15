# General libraries
# ===============
import numpy as np
import scipy
import glob
import os
import pandas as pd
from pathlib import Path

# Specific libraries
# ================
from data_preparation import load_data, prepare_data

from src.config import *
from tqdm import tqdm
import torch.optim as optim
import re
from transformers import BatchFeature
import matplotlib.pyplot as plt
import numpy as np
#from cross_domain_saliency_maps.torch_ig.cross_domain_integrated_gradients import TimeIG


def my_norm(a):
    ratio = 2/(np.max(a)-np.min(a)) 
    shift = (np.max(a)+np.min(a))/2 
    return (a - shift)*ratio

def min_max_norm(x, min, max):
    return 2 * (x - min) / (max - min) - 1
# ======================================================================================
#                      UTILS FOR CROSS-VALIDATION SPLIT AND DATA LOADING
# ======================================================================================
def divide_data_on_parts(data, labels, timesteps, filenames_list, parts=2):
    """
    Divide the list of audio samples (and corresponding breathing signals) into the specified number of parts.

    :param data: (np.ndarray) Loaded audio files as a NumPy array.
    :param labels: (np.ndarray) Loaded breathing ground truth as a NumPy array.
    :param timesteps: (np.ndarray) Timesteps corresponding to each label point.
    :param filenames_list: (list) List of filenames, where indexes match those in `data` and `labels`.
    :param parts: (int) Number of parts into which the data should be divided.

    :return list_of_parts: (list) A list of parts, where each part is a list containing the data, labels, timesteps, and filenames.
    """
    list_parts = []
    length_part = int(data.shape[0] / parts)
    start_point = 0

    for i in range(parts - 1):
        tmp_data = data[start_point:(start_point + length_part)]
        tmp_labels = labels[start_point:(start_point + length_part)]
        tmp_timesteps = timesteps[start_point:(start_point + length_part)]
        tmp_filenames_list = filenames_list[start_point:(start_point + length_part)]
        idx = 0

        list_parts.append((tmp_data, tmp_labels, tmp_timesteps, tmp_filenames_list))
        start_point += length_part

    tmp_data = data[start_point:]
    tmp_labels = labels[start_point:]
    tmp_timesteps = timesteps[start_point:]
    tmp_filenames_list = filenames_list[start_point:]
    idx = 0

    list_parts.append((tmp_data, tmp_labels, tmp_timesteps, tmp_filenames_list))

    return list_parts


def form_train_and_val_datasets(train_parts, dev_parts, fold_index):
    """
    Defines the validation and training sets for fold `fold_index`.

    :param train_parts: (list) Training data folds/parts.
    :param dev_parts: (list) Development data folds/parts.
    :param fold_index: (int) Index of the part to be used as the validation set.
        The index is applied to the concatenated list `[train_parts, dev_parts]`.

    :return: train_dataset: (list) Data used for training in fold `fold_index`.
    :return: val_dataset: (list) Data used for validation in
    """

    total = []
    for i in range(len(train_parts)):
        total.append(train_parts[i])
    for i in range(len(dev_parts)):
        total.append((dev_parts[i]))
    val_dataset = [total.pop(fold_index)]
    train_dataset = total
    return train_dataset, val_dataset


def extract_list_of_parts(list_of_parts):
    """
    Retrieves information from `list_of_parts`.

    :param list_of_parts: List of lists containing data, labels, timesteps, and filename dictionaries.

    :return: data: (np.array) Loaded audio files as a NumPy array.
    :return: labels: (np.array) Loaded breathing ground truth as a NumPy array.
    :return: timesteps: (np.array) Timesteps corresponding to each label point.
    :return: filenames_list: (list) List of filenames, where indexes match those in `data` and `labels`.
    """

    data = list_of_parts[0][0]
    labels = list_of_parts[0][1]
    timesteps = list_of_parts[0][2]
    filename_list = list_of_parts[0][3]
    for i in range(1, len(list_of_parts)):
        data = np.append(data, list_of_parts[i][0], axis=0)
        labels = np.append(labels, list_of_parts[i][1], axis=0)
        timesteps = np.append(timesteps, list_of_parts[i][2], axis=0)
        filename_list = np.append(filename_list, list_of_parts[i][3], axis=0)

    result_data, result_labels, result_timesteps = reshaping_data_for_model(data, labels, timesteps)

    return result_data, result_labels, result_timesteps, filename_list


def get_parts(path_to_data, path_to_labels, set_name, annotated, length_sequence, step_sequence, data_parts):
    """
    Divides the given data into parts and loads the information.

    :param path_to_data: (str) Path to the audio data.
    :param path_to_labels: (str) Path to the `.csv` file containing breathing signal values.
    :param set_name: (str) Name of the dataset (`train` or `dev`).
    :param annotated: (bool) `True` if the dataset includes ground truth labels.
    :param length_sequence: (int) Window length for audio analysis.
    :param step_sequence: (int) Step size for the sliding window.
    :param data_parts: (int) Number of parts into which the data will be divided.

    :return: (list) A list of parts, where each part is a list containing the data, labels, timesteps, and filenames.
    """

    data, labels, filename_dict, frame_rate = load_data(path_to_data, path_to_labels, set_name)

    prepared_data, prepared_labels, prepared_timesteps, filename_list = prepare_data(data, labels, filename_dict,
                                                                                     frame_rate,
                                                                                     annotated,
                                                                                     length_sequence,
                                                                                     step_sequence)

    # divide train data on parts
    parts = divide_data_on_parts(prepared_data, prepared_labels, prepared_timesteps,
                                 parts=data_parts, filenames_list=filename_list)

    return parts


def load_parts(train_parts, devel_parts, index_of_part):
    """
    Creates training and validation datasets from the provided parts.

    :param train_parts: (list) List of training data parts.
    :param devel_parts: (list) List of development data parts.
    :param index_of_part: (int) Index used to split the training and validation sets for cross-validation.

    :return: tuple containing:
        - Training data, training labels, training timesteps, training filenames
        - Validation data, validation labels, validation timesteps, validation filenames
    """

    # form train and validation datasets from train and development parts of data
    train_dataset, val_dataset = form_train_and_val_datasets(train_parts, devel_parts,
                                                             fold_index=index_of_part)
    # unpacking data from train_dataset to make it readable for keras
    train_d, train_lbs, train_timesteps, train_filenames_list = extract_list_of_parts(list_of_parts=train_dataset)
    # unpacking data from val_dataset to make it readable for keras
    val_d, val_lbs, val_timesteps, val_filenames_list = extract_list_of_parts(list_of_parts=val_dataset)

    return train_d, train_lbs, train_timesteps, train_filenames_list, val_d, val_lbs, val_timesteps, val_filenames_list


def concatenate_prediction(predicted_values, labels_timesteps, filenames_list,
                           columns_for_real_labels=['filename', 'timeFrame', 'upper_belt']):
    
    """
    Concatenate the predictions by averaging predictions per timestep. Since we work with a sliding window, there are 
    multiple predictions per timestep.

    :param predicted_values: predicted values for given set
    :param labels_timesteps: corresponding timesteps
    :param filenames_list: corresponding filenames
    :param columns_for_real_labels: columns for the ground truth labels


    :return result_predicted_values: Dataframe with concatenated results


    """
    predicted_values = np.asarray(predicted_values).reshape(labels_timesteps.shape)
    result_predicted_values = pd.DataFrame(columns=columns_for_real_labels, dtype='float32')
    result_predicted_values['filename'] = result_predicted_values['filename'].astype('str')
    
    # for single file analysis
    if len(filenames_list) != 1:
        filenames_list = np.asarray(filenames_list).reshape(labels_timesteps.shape)

        if len(predicted_values) == 1:
            predicted_values = [predicted_values]
            labels_timesteps = [labels_timesteps]

        for instance_idx in range(predicted_values.shape[0]):
            predicted_values_tmp = predicted_values[instance_idx].reshape((-1, 1))
            timesteps_labels_tmp = labels_timesteps[instance_idx].reshape((-1, 1))
            filenames_tmp = filenames_list[instance_idx].reshape((-1, 1))
            tmp = pd.DataFrame(columns=['filename', 'timeFrame', 'upper_belt'],
                            data=np.concatenate((filenames_tmp, timesteps_labels_tmp, predicted_values_tmp), axis=1))
            result_predicted_values = pd.concat([result_predicted_values, tmp.copy(deep=True)], ignore_index=True)
    else:  
        for instance_idx in range(predicted_values.shape[0]):
            predicted_values_tmp = predicted_values[instance_idx].reshape((-1, 1))
            timesteps_labels_tmp = labels_timesteps[instance_idx].reshape((-1, 1))
            tmp = pd.DataFrame(columns=['timeFrame', 'upper_belt'],
                            data=np.concatenate((timesteps_labels_tmp, predicted_values_tmp), axis=1))
     
            result_predicted_values = pd.concat([result_predicted_values, tmp.copy(deep=True)], ignore_index=True)

    result_predicted_values['timeFrame'] = result_predicted_values['timeFrame'].astype('float32')
    result_predicted_values['upper_belt'] = result_predicted_values['upper_belt'].astype('float32')
    if len(filenames_list) == 1:
        result_predicted_values['filename'] = filenames_list[0]
        result_predicted_values = result_predicted_values.groupby(['timeFrame'], as_index=False)[
            'upper_belt'].mean()  # deals with overlap
    else:
        result_predicted_values = result_predicted_values.groupby(['filename', 'timeFrame'], as_index=False)[
        'upper_belt'].mean()
    return result_predicted_values

# ======================================================================================
#                               UTILS FOR MODEL TRAINING
# ======================================================================================

def correlation_coefficient_loss_torch(y_true, y_pred):
    """
    Computes the PCC loss for PyTorch models.

    :param y_true: (torch.Tensor) Concatenated ground truth breathing labels.
    :param y_pred: (torch.Tensor) Concatenated predicted breathing labels.

    :return: (torch.Tensor) `1 - PCC`.
    """

    if y_true.ndim == 1:
        x = y_true.unsqueeze(0)  # (1, T)
        y = y_pred.unsqueeze(0)  # (1, T)

    else:
        x = y_true
        y = y_pred
    mx = torch.mean(x, axis=1, keepdims=True)
    my = torch.mean(y, axis=1, keepdims=True)
    xm, ym = x - mx, y - my
    r_num = torch.sum(torch.multiply(xm, ym), axis=1)
    sum_square_x = torch.sum(torch.square(xm), axis=1)
    sum_square_y = torch.sum(torch.square(ym), axis=1)
    sqrt_x = torch.sqrt(sum_square_x)
    sqrt_y = torch.sqrt(sum_square_y)
    r_den = torch.multiply(sqrt_x, sqrt_y)

    # Calculate PCC
    r = torch.divide(r_num, r_den)
    r = torch.mean(r)

    # Handle NaN values
    r = torch.where(torch.isnan(r), torch.zeros_like(r), r)
    return 1 - r


def combined_loss_torch(y_true, y_pred):
    """
    Combines PCC and RMSE of standardized signals.
    Loss = 1 - PCC + RMSE(z-score(y_true), z-score(y_pred))
    """
    if y_true.ndim == 1:
        y_true = y_true.unsqueeze(0)
        y_pred = y_pred.unsqueeze(0)

    # Standardize each signal (per row)
    y_true_std = (y_true - torch.mean(y_true, dim=1, keepdim=True)) / (torch.std(y_true, dim=1, keepdim=True) + 1e-8)
    y_pred_std = (y_pred - torch.mean(y_pred, dim=1, keepdim=True)) / (torch.std(y_pred, dim=1, keepdim=True) + 1e-8)

    # PCC
    xm, ym = y_true_std, y_pred_std
    r_num = torch.sum(xm * ym, dim=1)
    r_den = torch.sqrt(torch.sum(xm * xm, dim=1)) * torch.sqrt(torch.sum(ym * ym, dim=1))
    r = torch.mean(r_num / r_den)
    r = torch.where(torch.isnan(r), torch.zeros_like(r), r)

    # RMSE on standardized signals
    rmse = torch.sqrt(torch.mean((y_true_std - y_pred_std) ** 2))

    loss = (1 - r) + rmse
    return loss


def initialize_model(exp_parameters, val_lbs):
    """
    Initializes the model, optimizer, and scheduler based on experiment parameters.

    :param exp_parameters: (dict) Dictionary containing experiment parameters.
    :param val_lbs: (np.ndarray) Validation labels for determining output size.
    :return: model, optimizer, scheduler
    """
    output_size = val_lbs.shape[-1]
    if 'n_layers' in exp_parameters.keys():
        model = exp_parameters['model_class'](output_size, exp_parameters['window_size'], exp_parameters['n_layers']).to(device)
    else:
        model = exp_parameters['model_class'](output_size, exp_parameters['window_size']).to(device)
    
    optimizer = None
    scheduler = None

    if exp_parameters['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=exp_parameters['learning_rate'])
    elif exp_parameters['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=exp_parameters['learning_rate'], momentum=0.9)
    elif exp_parameters['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=exp_parameters['learning_rate'])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=1, eta_min=1e-5
        )

    return model, optimizer, scheduler

def initialize_or_load_model(exp_parameters, val_lbs, index_of_part):
    """
    Initializes a new model or loads the best trained model if available.

    :param exp_parameters: (dict) Experiment parameters.
    :param val_lbs: (np.ndarray) Validation labels for determining output size.
    :param index_of_part: (int) Dataset split index.
    :return: model, optimizer, scheduler, trained_epochs (int)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_size = val_lbs.shape[-1]

    # Build model folder name pattern
    model_folder_name = 'best_%s_window_%s_step_%s_batch_size_%s_epochs_%s_loss_%s_optimizer_%s_learning_rate_%s' % (
        exp_parameters['model_name'],
        str(exp_parameters['window_size']).replace('.', '_'),
        str(exp_parameters['step_size']).replace('.', '_'),
        exp_parameters['batch_size'],
        exp_parameters['epochs'],
        exp_parameters['loss'],
        exp_parameters['optimizer'],
        str(exp_parameters['learning_rate']).replace('.', '_')
    )

    model_path_pattern = f"{MODELS_FOLDER}/{model_folder_name}/best_model_weights_idx_of_part_{index_of_part}_epoch_*"
    matching_files = glob.glob(model_path_pattern)

    if matching_files:
        # Load the most recent or first matching file
        best_model_path = matching_files[0]
        # Extract epochs from filename
        epoch_match = re.search(r'epoch_(\d+)', os.path.basename(best_model_path))
        trained_epochs = int(epoch_match.group(1)) if epoch_match else None

        # Initialize model and load weights
        if 'n_layers' in exp_parameters.keys():
            model = exp_parameters['model_class'](output_size, exp_parameters['window_size'], exp_parameters['n_layers']).to(device)
        elif exp_parameters['model_class'] == BaseModel:
            model = exp_parameters['model_class'](output_size, exp_parameters['window_size'], exp_parameters['wav_model']).to(device)
        else:
            model = exp_parameters['model_class'](output_size, exp_parameters['window_size']).to(device)

        model.load_state_dict(torch.load(best_model_path, weights_only=False), strict=False)

        
    else:
        # No pre-trained model: initialize new model
        trained_epochs = 0
        if 'n_layers' in exp_parameters.keys():
            model = exp_parameters['model_class'](output_size, exp_parameters['window_size'], exp_parameters['n_layers']).to(device)
        
        elif exp_parameters['model_class'] == BaseModel:
            model = exp_parameters['model_class'](output_size, exp_parameters['window_size'], exp_parameters['wav_model']).to(device)
    
        else:
            model = exp_parameters['model_class'](output_size, exp_parameters['window_size']).to(device)

    optimizer = None
    scheduler = None
    if exp_parameters['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=exp_parameters['learning_rate'])
    elif exp_parameters['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=exp_parameters['learning_rate'], momentum=0.9)
    elif exp_parameters['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=exp_parameters['learning_rate'])
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=1, eta_min=1e-5
        )

    return model, optimizer, scheduler, trained_epochs

def train_step(model, train_loader, optimizer, loss_fn):
    """
    Performs one epoch of training.

    :param model: (nn.Module) The PyTorch model.
    :param train_loader: (DataLoader) DataLoader for training data.
    :param optimizer: (torch optimizer) Optimizer for model training.
    :param loss_fn: Loss function.

    :return: (float) Average training loss.
    """
    model.train()
    train_loss = 0.0
    progress_bar = tqdm(train_loader, desc="Training")

    for batch_audio, batch_audio_emb, batch_lbs, batch_speaker_emb, filename in progress_bar:
        optimizer.zero_grad()
        batch_audio = batch_audio.to(device)
        batch_lbs = batch_lbs.to(device)
        
        if batch_audio_emb[0] != []:
            batch_audio_emb = batch_audio_emb.to(device)
        if batch_speaker_emb[0] != None:
            batch_speaker_emb = batch_speaker_emb.to(device)

        outputs = model(batch_audio, batch_audio_emb, speaker_embs = batch_speaker_emb)
        
        batch_size = batch_lbs.shape[0]

        loss = loss_fn(outputs, batch_lbs)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        progress_bar.set_postfix({'train_loss': f'{train_loss / (progress_bar.n + 1):.4f}'})

    return train_loss / len(train_loader)


def validate_model(model, val_loader, loss_fn, val_timesteps, val_filenames_list, concatenated_ground_truth, annotated = True):
    """
    Performs model validation.

    :param model: (nn.Module) The PyTorch model.
    :param val_loader: (DataLoader) DataLoader for validation data.
    :param loss_fn: Loss function.
    :param val_timesteps: (np.array) Numpy array of validation timesteps.
    :param val_filenames_list: (list) List of validation filenames.
    :param concatenated_ground_truth: (DataFrame) Ground truth data for PCC calculation.

    :return: Tuple (validation loss, Pearson coefficient, concatenated predicted labels).
    """
    model.eval()
    val_loss = 0.0
    predicted_labels = []

    with torch.no_grad():
        for data in val_loader:
            batch_audio, batch_audio_emb, batch_lbs, batch_speaker_emb, filename = data
       
            batch_audio = batch_audio.to(device)
            batch_lbs = batch_lbs.to(device)
            if batch_audio_emb[0] != []:
                batch_audio_emb = batch_audio_emb.to(device)
            if batch_speaker_emb[0] != None:
                batch_speaker_emb = batch_speaker_emb.to(device)
            outputs = model(batch_audio, batch_audio_emb, speaker_embs = batch_speaker_emb)

            val_output_loss = loss_fn(outputs, batch_lbs)
            val_loss += val_output_loss.item()
            predicted_labels.extend(outputs.cpu().numpy())
    concatenated_predicted_labels = concatenate_prediction(predicted_labels, val_timesteps, val_filenames_list)['upper_belt'].values
    # Here we use my_norm instead of min_max_norm because this function will be run during training, 
    # hence getting the min and max of the predictions in the training set would make the training more
    # computationally expensive
    concatenated_predicted_labels_norm = my_norm(concatenated_predicted_labels)
    if annotated:
        if isinstance(concatenated_ground_truth, pd.DataFrame):
            prc_coef = scipy.stats.pearsonr(concatenated_ground_truth['upper_belt'].values,
                                        concatenated_predicted_labels)
        else:
            
            prc_coef = scipy.stats.pearsonr(concatenated_ground_truth,
                                concatenated_predicted_labels)
    else:
        prc_coef = [0]

    return val_loss / len(val_loader), prc_coef[0], concatenated_predicted_labels_norm, concatenated_predicted_labels, predicted_labels


def save_best_model(model, path_to_save_best_model, index_of_part, epoch):
    """
    Saves the best model and removes previous best models.

    :param model: The PyTorch model.
    :param path_to_save_best_model: Path to save the best model.
    :param index_of_part: Cross-validation fold index.
    :param epoch: Current epoch number.
    """
    for file in glob.glob(f"{path_to_save_best_model}/best_model_weights_idx_of_part_{index_of_part}_epoch_*"):
        os.remove(file)

    torch.save(model.state_dict(),
               f"{path_to_save_best_model}/best_model_weights_idx_of_part_{index_of_part}_epoch_{epoch}")


def reshaping_data_for_model(data, labels, timesteps):
    """
    Reshapes the input data, labels, and timesteps by removing the first dimension, which combines all the windows for each sample.

    :param data: (np.array) The windowed speech audio to be reshaped.
    :param labels: (np.array) The corresponding windowed labels to be reshaped.
    :param timesteps: (np.array) The windowed timesteps to be reshaped.

    :return result_data: (np.array) Reshaped data.
    :return labels: (np.array) Reshaped labels.
    :return timesteps: (np.array) Reshaped timesteps.
    """
    result_data = data.reshape((-1, data.shape[2]))
    result_labels = labels.reshape((-1, labels.shape[2]))
    result_timesteps = timesteps.reshape((-1, timesteps.shape[2]))
    return result_data, result_labels, result_timesteps


def load_best_model(model, exp_parameters, index_of_part):
    """
    Loads the best model weights for evaluation.

    :param model: (nn.Module) PyTorch model instance.
    :param exp_parameters: (dict) Dictionary containing experiment parameters.
    :param index_of_part: (int) Index of the dataset split part.

    :return model: (nn.Module) model with loaded weights.
    """
    model_folder_name = 'best_%s_window_%s_step_%s_batch_size_%s_epochs_%s_loss_%s_optimizer_%s_learning_rate_%s' % (
        exp_parameters['model_name'], str(exp_parameters['window_size']).replace('.', '_'),
        str(exp_parameters['step_size']).replace('.', '_'), exp_parameters['batch_size'], exp_parameters['epochs'],
        exp_parameters['loss'], exp_parameters['optimizer'], str(exp_parameters['learning_rate']).replace('.', '_')
    )

    best_model_path = glob.glob(
        f"{MODELS_FOLDER}/{model_folder_name}/best_model_weights_idx_of_part_{index_of_part}_epoch_*"
    )[0]

    model.load_state_dict(torch.load(best_model_path, weights_only=False), strict=False)
    model.eval()
    return model

