import sys
import os
project_root = os.path.dirname('..')
sys.path.append(os.getcwd())
from scipy.stats import pearsonr
from cross_validation_train import *



from src.config import *



def eval_test(path_to_output_predictions, path_to_data, path_to_labels, exp_parameters):
    """
    Evaluate trained models on the test set.

    :param path_to_output_predictions: (str) path to the folder to save breathing predictions
    :param path_to_data: (str) path to test data
    :param path_to_labels: (str) path to test labels
    :param exp_parameters: (dict) experimental parameters of the model to evaluate
    
    :return: None
    """

    if os.path.exists(path_to_output_predictions):
        pass
    else:
        os.mkdir(path_to_output_predictions)

    test_data, test_labels, test_dict, frame_rate = load_data(path_to_data, path_to_labels, 'test')

    length_sequence = int(exp_parameters['window_size'] * 16000)  # 16000 is the sample rate
    step_sequence = int(exp_parameters['step_size'] * 16000)  # int(2 / 5 * length_sequence)
    test_d, test_lbs, test_timesteps, test_filenames_list = prepare_data(test_data, test_labels, test_dict, frame_rate,
                                                                         'ANNOTATED', length_sequence, step_sequence)
    test_d = np.vstack(test_d)
    test_lbs = np.vstack(test_lbs)
    test_timesteps = np.vstack(test_timesteps)
    test_filenames_list = np.vstack(test_filenames_list)
    concatenated_ground_truth = concatenate_prediction(test_lbs, test_timesteps,
                                                       test_filenames_list)
    batch_size = exp_parameters['batch_size']
    test_dataset = UCL_SBM_Dataset(test_d, test_lbs, test_timesteps, test_filenames_list, exp_parameters)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1, collate_fn=test_dataset.collate_fn)

    all_predictions = []
    all_predictions_norm = []
    for index_of_part in range(4):
        # Create and initialize model
        pcc, concat_predictions, concat_predictions_norm, predictions = eval_model(test_loader, test_lbs, test_timesteps,
                                      test_filenames_list, concatenated_ground_truth, exp_parameters,
                                      index_of_part)

        train_info = pd.read_csv(f'{path_to_output_predictions}/train_info_{index_of_part}.csv')
        all_predictions.append(predictions)
        all_predictions_norm.append(min_max_norm(predictions, train_info['min'].values[0], train_info['max'].values[0]))

        print(f'FOLD {index_of_part}: {pcc}')

    mean_prediction = np.sum(all_predictions, 0) / 4
    mean_prediction_norm = np.sum(all_predictions_norm, 0) / 4
    concatenated_predictions = concatenate_prediction(mean_prediction, test_timesteps,
                                                      test_filenames_list)
    concatenated_predictions_norm = concatenate_prediction(mean_prediction_norm, test_timesteps,
                                                      test_filenames_list)
    
    
    final_test_pcc = scipy.stats.pearsonr(concatenated_ground_truth['upper_belt'].values,
                                          concatenated_predictions['upper_belt'].values)
    final_norm_test_pcc = scipy.stats.pearsonr(concatenated_ground_truth['upper_belt'].values,
                                          concatenated_predictions_norm['upper_belt'].values)
    print('Test PCC:', final_test_pcc[0])
    print('Test Norm PCC:', final_norm_test_pcc[0])



def main():

    list_parameters = [WAVLM_CNN_16, WAVLM_CNN_FiLM]

    for exp_parameters in list_parameters:

        model_folder_name = 'best_%s_window_%s_step_%s_batch_size_%s_epochs_%s_loss_%s_optimizer_%s_learning_rate_%s' % (
            exp_parameters['model_name'], str(exp_parameters['window_size']).replace('.', '_'),
            str(exp_parameters['step_size']).replace('.', '_'), exp_parameters['batch_size'], exp_parameters['epochs'],
            exp_parameters['loss'], exp_parameters['optimizer'], str(exp_parameters['learning_rate']).replace('.', '_'))

        print("MODEL - %s \n -window: %d \n -step %f \n -batch_size: %d \n -epochs: %d \n -loss %s  \n -optimizer: %s "
              "\n -learning_rate:%f" % (exp_parameters['model_name'], exp_parameters['window_size'],
                                        exp_parameters['step_size'], exp_parameters['batch_size'],
                                        exp_parameters['epochs'], exp_parameters['loss'], exp_parameters['optimizer'],
                                        exp_parameters['learning_rate']))
        OUTPUT = 'data/output'
        path_to_output_predictions = os.path.join(OUTPUT, model_folder_name)

        if not os.path.exists(path_to_output_predictions):
            os.mkdir(path_to_output_predictions)

        eval_test(path_to_output_predictions, AUDIO_INTERSPEECH, BREATH_INTERSPEECH, exp_parameters)


if __name__ == "__main__":
    main()
