import os
import sys

project_root = os.path.dirname('..')
sys.path.append(project_root)

from src.config import *
from cross_validation_train import train_and_eval_cv_model
import sys

if __name__ == "__main__":

    list_parameters = [WAVLM_CNN_0, WAVLM_CNN_16, WAVLM_CNN_EGEMAPS_16, WAVLM_CNN_WESPEAKER_16, 
    WAVLM_ATTENTION, WAVLM_ATTENTION_EGEMAPS, WAVLM_ATTENTION_WESPEAKER]
    
    for exp_parameters in list_parameters:
        for mode in ['eval']: 
            for exp_parameters in list_parameters:
                print(
                    "MODEL - %s \n -window: %d \n -step %f \n -batch_size: %d \n -epochs: %d \n -loss %s  \n -optimizer: "
                    "%s \n -learning_rate:%f" % (
                        exp_parameters['model_name'], exp_parameters['window_size'],
                        exp_parameters['step_size'], exp_parameters['batch_size'], exp_parameters['epochs'],
                        exp_parameters['loss'], exp_parameters['optimizer'], exp_parameters['learning_rate']))


                model_folder_name = '%s_window_%s_step_%s_batch_size_%s_epochs_%s_loss_%s_optimizer_%s_learning_rate_%s' % (
                    exp_parameters['model_name'], str(exp_parameters['window_size']).replace('.', '_'),
                    str(exp_parameters['step_size']).replace('.', '_'), exp_parameters['batch_size'],
                    exp_parameters['epochs'],
                    exp_parameters['loss'], exp_parameters['optimizer'],
                    str(exp_parameters['learning_rate']).replace('.', '_'))

                # please, specify paths to data and other parameters
                path_to_save_model =  MODELS_FOLDER + 'best_%s' % model_folder_name
                if not os.path.exists(path_to_save_model):
                    os.mkdir(path_to_save_model)
                    print(model_folder_name)
                path_to_output = OUTPUT_FOLDER + 'best_%s' % model_folder_name
                if not os.path.exists(path_to_output):
                    os.mkdir(path_to_output)
                    
                train_and_eval_cv_model(exp_parameters,
                                        data_parts=2,
                                        path_to_save_models=path_to_save_model,
                                        path_to_train_data=AUDIO_INTERSPEECH,
                                        path_to_train_labels=BREATH_INTERSPEECH,
                                        path_to_devel_data=AUDIO_INTERSPEECH,
                                        path_to_devel_labels
                                        =BREATH_INTERSPEECH,
                                        mode=mode, start_index= 0)
