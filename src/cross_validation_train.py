from utils import *
from torch.utils.data import Dataset, DataLoader
from data_preparation import *
import torch.nn as nn


def train_model(train_loader, val_loader, val_lbs, val_timesteps,
                val_filenames_list, concatenated_ground_truth, exp_parameters,
                path_to_save_best_model, index_of_part):
    """
    Trains a model using the PyTorch framework for the fold specified by `index_of_part`.

    :param train_loader: (DataLoader) Training data.
    :param val_loader: (DataLoader) Validation data.
    :param val_lbs: (np.array) Validation labels.
    :param val_timesteps: (np.array) Validation timesteps.
    :param val_filenames_list: (list) List of filenames corresponding to each timestep in the validation set.
    :param concatenated_ground_truth: (DataFrame) Concatenated ground truth for the validation set.
    :param exp_parameters: (dict) Experimental parameters; keys vary depending on the model type.
    :param path_to_save_best_model: (str) Path to save the best model.
    :param index_of_part: (int) Cross-validation fold index.

    :return: None
    """

    model, optimizer, scheduler, start_epochs = initialize_or_load_model(exp_parameters, val_lbs, index_of_part)

    if exp_parameters['loss'] == 'PCC':
        loss_fn = correlation_coefficient_loss_torch
    elif exp_parameters['loss'] == 'MSE':
        loss_fn = nn.MSELoss()
    
    elif exp_parameters['loss'] == 'Combined':
        loss_fn = combined_loss_torch

    best_result = -float('inf')
    early_stopping_counter = 0

    for epoch in range(start_epochs, exp_parameters['epochs']):
        train_loss = train_step(model, train_loader, optimizer, loss_fn)

        # Validate every 2 epochs to speed up training
        if epoch % 2 == 0:
            val_loss, prc_coef, _, _, _= validate_model(model, val_loader, loss_fn, val_timesteps,
                                                   val_filenames_list, concatenated_ground_truth)

            print(f'Epoch {epoch + 1}, Pearson Coefficient: {prc_coef:.4f}, Validation Loss: {val_loss:.4f}')

            if prc_coef > best_result:
                best_result = prc_coef
                early_stopping_counter = 0
                save_best_model(model, path_to_save_best_model, index_of_part, epoch + 1)
            
            else:
                early_stopping_counter += 1
                print(f"Early stopping counter: {early_stopping_counter}")

            if 'patience' in exp_parameters.keys():
                if early_stopping_counter >= exp_parameters['patience']:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    
                    break

        if scheduler:
            scheduler.step()


def eval_model(val_loader, val_lbs, val_timesteps, val_filenames_list, concatenated_ground_truth, exp_parameters,
                     index_of_part, output_path=os.getcwd(), annotated = True):
    """
    Evaluate PyTorch model using the UCL_SBM_Dataset validation data loader.

    :param val_loader: (DataLoader) UCL_SBM_Dataset DataLoader for the validation data.
    :param val_lbs: (np.ndarray) Validation breathing labels.
    :param val_timesteps: (list) List of validation timesteps.
    :param val_filenames_list: (list) List of validation filenames.
    :param concatenated_ground_truth: (pd.DataFrame) concatenated ground truth breathing data for PCC calculation.
    :param exp_parameters: (dict) Dictionary containing experiment parameters.
    :param index_of_part: (int) Index of the current part of the dataset.
   

    :return prc_coef: (float) Pearson correlation coefficient of the predicted concatenated labels and the concatenated_ground_truth.
    :return concatenated_predicted_labels: (DataFrame) Predicted concatenated labels.
    :return concatenated_predicted_labels_norm: (DataFrame) Normalized predicted concatenated labels.
    :return predicted_labels: (DataFrame) Non-concatenated labels 
    """

    model, _, _ = initialize_model(exp_parameters, val_lbs)
    model = load_best_model(model, exp_parameters, index_of_part)

    # Set loss
    if exp_parameters['loss'] == 'PCC':
        loss_fn = correlation_coefficient_loss_torch
    elif exp_parameters['loss'] == 'MSE':
        loss_fn = torch.nn.MSELoss()
    else:
        loss_fn = combined_loss_torch

    # Run validation
    _, prc_coef, concatenated_predicted_labels_norm, concatenated_predicted_labels, predictions = validate_model(
        model, val_loader, loss_fn=loss_fn, val_timesteps=val_timesteps, val_filenames_list=val_filenames_list,
        concatenated_ground_truth=concatenated_ground_truth, annotated = annotated)

    # Save fold results
    if isinstance(concatenated_ground_truth, pd.DataFrame):
        fold_results = concatenated_ground_truth.copy()
        fold_results['prediction'] = concatenated_predicted_labels
        fold_results['norm_prediction'] = concatenated_predicted_labels_norm
        fold_results.to_csv(f'{output_path}/results_fold_{index_of_part}.csv', index=False)

    return prc_coef, concatenated_predicted_labels, concatenated_predicted_labels_norm, predictions

def get_min_max_train(train_loader, train_lbs, train_timesteps, train_filenames_list, index_of_part, exp_parameters, output_path):
    """
    Get minimum and maximum value for the breathing prediction on the training set, using the model trained in that data. 
    This values will be used to normalize the predictions.

    :param train_loader: (DataLoader) UCL_SBM_Dataset DataLoader for the training data.
    :param train_lbs: (np.ndarray) Training breathing labels.
    :param train_timesteps: (list) List of training timesteps.
    :param train_filenames_list: (list) List of training filenames.
    :param index_of_part: (int) Index of the current part of the dataset.
    :param exp_parameters: (dict) Dictionary containing experiment parameters.
    :param output_path: (str) Output folder, used to save the information from the predictions on the training set

    :return: None
    """
    if exp_parameters['loss'] == 'PCC':
        loss_fn = correlation_coefficient_loss_torch
    elif exp_parameters['loss'] == 'MSE':
        loss_fn = nn.MSELoss()
    
    elif exp_parameters['loss'] == 'Combined':
        loss_fn = combined_loss_torch

    model, _, _ = initialize_model(exp_parameters, train_lbs)
    model = load_best_model(model, exp_parameters, index_of_part)  

    model.eval()
    val_loss = 0.0
    predicted_labels = []

    with torch.no_grad():
        for data in train_loader:
            batch_audio, batch_audio_emb, batch_lbs, batch_speaker_emb, filename = data
       
            batch_audio = batch_audio.to(device)
            batch_lbs = batch_lbs.to(device)
            if batch_audio_emb[0] != []:
                batch_audio_emb = batch_audio_emb.to(device)
            if batch_speaker_emb[0] != None:
                batch_speaker_emb = batch_speaker_emb.to(device)
            outputs = model(batch_audio, batch_audio_emb, speaker_embs = batch_speaker_emb)

            predicted_labels.extend(outputs.cpu().numpy())

    concatenated_predicted_labels = concatenate_prediction(predicted_labels, train_timesteps, train_filenames_list)['upper_belt'].values
    train_info = {'max': np.max(concatenated_predicted_labels), 
            'min': np.min(concatenated_predicted_labels), 
            'mean':np.mean(concatenated_predicted_labels),
            'std':np.std(concatenated_predicted_labels)}

    train_info_df = pd.DataFrame(train_info, index = [exp_parameters['model_name']])
    train_info_df.to_csv(f'{output_path}/train_info_{index_of_part}.csv')

#####################################################################################################
#            
#                                           GENERAL
#
#####################################################################################################


def train_and_eval_cv_model(exp_parameters,
                            data_parts,
                            path_to_save_models,
                            path_to_train_data,
                            path_to_train_labels,
                            path_to_devel_data,
                            path_to_devel_labels, mode, start_index = 0):
    """
    Train and evaluate with cross-validation models using the provided data paths and experiment parameters.

    :param exp_parameters: (dict) Dictionary containing experiment parameters.
    :param data_parts: (int) Number of data parts for cross-validation.
    :param path_to_save_models: (str) Path to save the trained models.
    :param path_to_train_data: (str) Path to the speech audio data.
    :param path_to_train_labels: (str) Path to the breathing data csv file.
    :param path_to_devel_data: (str) Path to the speech audio data.
    :param path_to_devel_labels: (str) Path to the breathing data csv file.
    :param mode: (str) mode 'train' or 'eval'.
    :param start_index: (int) fold to start cross_validation

    :return: None
    """

    # Window parameters
    length_sequence = int(exp_parameters['window_size'] * 16000)  # 16000 is the sample rate
    step_sequence = int(exp_parameters['step_size'] * 16000)  # int(2 / 5 * length_sequence)

    # Create folders to save models
    path_to_save_best_model = path_to_save_models
    if not os.path.exists(path_to_save_best_model):
        os.mkdir(path_to_save_best_model)

    # Divide parts
    train_parts = get_parts(path_to_train_data, path_to_train_labels, 'train', True, length_sequence,
                            step_sequence, data_parts)
    devel_parts = get_parts(path_to_devel_data, path_to_devel_labels, 'devel', True, length_sequence,
                            step_sequence, data_parts)

    all_pcc = []


    for index_of_part in range(start_index, len(train_parts) + len(devel_parts)):
        print('FOLD %d \n===========================' % index_of_part)
        train_d, train_lbs, train_timesteps, train_filenames_list, val_d, val_lbs, val_timesteps, val_filenames_list = load_parts(
            train_parts, devel_parts, index_of_part)
        # Get combined ground truths for validation set
        concatenated_ground_truth = concatenate_prediction(val_lbs, val_timesteps,
                                                        val_filenames_list)

        batch_size = exp_parameters['batch_size']
        val_dataset = UCL_SBM_Dataset(val_d, val_lbs,  val_timesteps, val_filenames_list, exp_parameters)

        # Create DataLoaders
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=val_dataset.collate_fn, num_workers= 8)

        train_dataset = UCL_SBM_Dataset(train_d, train_lbs, train_timesteps, train_filenames_list, exp_parameters)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                collate_fn=train_dataset.collate_fn, num_workers= 8)

        
        # Create and initialize model
        if mode == 'train':
            train_model(train_loader, val_loader, val_lbs, val_timesteps,
                        val_filenames_list, concatenated_ground_truth, exp_parameters,
                        path_to_save_best_model, index_of_part)

        if mode == 'eval':
            pcc = eval_model(val_loader, val_lbs, val_timesteps,
                                    val_filenames_list, concatenated_ground_truth, exp_parameters,
                                    index_of_part, output_path= path_to_save_models.replace(MODELS_FOLDER, OUTPUT_FOLDER))[0]
            
            all_pcc.append(pcc)
            print(f'PCC = {pcc}')
            get_min_max_train(train_loader, train_lbs, train_timesteps, train_filenames_list, index_of_part, exp_parameters, path_to_save_models.replace(MODELS_FOLDER, OUTPUT_FOLDER))
    
    if mode == 'eval':
        print('Mean PCC is: %.3f' % np.mean(all_pcc))
