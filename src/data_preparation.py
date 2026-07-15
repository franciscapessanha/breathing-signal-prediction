
import pandas as pd
from scipy.io import wavfile
import torch
from torch.utils.data import Dataset
import numpy as np
import librosa
import glob
from transformers import Wav2Vec2Processor
import wespeaker 
from src.config import *
from pathlib import Path
import re 
def load_data(path_to_data, path_to_labels, set_name):
    """
    Load data and corresponding labels in a given path.

    :param path_to_data: path to the audio files (.wav)
    :param path_to_labels: path to the breathing signal files (.csv)
    :param set_name: set label (train, dev or test)

    :return: result_data: numpy array with loaded audio files
    :return: labels: dataframe with labels corresponding to the given set.

    If the set is test (so no known labels) the label with be zero.
    :return: filename_dict: dictionary with the filenames. Indexes are the same as the ones for result_data and labels.
    :return: frame_rate: audio framerate in Hz
    """

    # Load labels
    labels = pd.read_csv(path_to_labels + 'all_labels.csv',  low_memory= False, sep=',') # error about mix types is due to the test set including ? as upper belt values
    labels = labels.loc[labels['filename'].str.contains(set_name)]

    labels['upper_belt'] = labels['upper_belt'].astype('float32')


    # Load data
    files = np.unique(labels['filename'])
    filename_dict = {}
    for i in range(len(files)):
        frame_rate, data = wavfile.read(path_to_data + files[i])
        filename_dict[i] = files[i]
        if i == 0:
            result_data = np.zeros(shape=(np.unique(labels['filename']).shape[0], data.shape[0]))
        result_data[i] = data

    return result_data, labels, filename_dict, frame_rate


def get_number_of_windows(length_sequence, window_size, step):
    """
    Gets number of windows needed given a certain sequence, window size and step.

    :param length_sequence: length of the audio sequence
    :param window_size: window size (number of datapoints per window)
    :param step: window step

    :return: counter: number of windows
    """
    start_idx = 0
    counter = 0
    while start_idx + window_size < length_sequence: 
        counter += 1
        start_idx += step
    
    # Add an additional window if there are remaining points
    if start_idx < length_sequence:
        counter += 1

    return counter


def prepare_data(data, labels, filename_dict, frame_rate, annotated, size_window, step_for_window):
    """
    Prepare data for the model. The output will split the data into smaller windows with the input size required for the model.

    :param data: numpy array with loaded data
    :param labels: dataframe with loaded labels
    :param filename_dict: dictionary with corresponding filenames
    :param frame_rate: audio frame rate
    :param annotated: true if there are labels for the data provided
    :param size_window: window size
    :param step_for_window: window step

    :return: windowed_data: data split into windows
    :return: windowed_labels: labels split into windows
    :return: windowed_labels_timesteps: corresponding timesteps for the windowed labels
    :return: all_filenames: list containing the filenames for each datapoint in windowed_labels
    """

    label_rate = 25  # The breathing signal has a sample rate of 25 Hz

    # Crop audio with sliding window method.Each window will be a sample.
    num_windows = get_number_of_windows(data.shape[1], size_window, step_for_window)
    windowed_data = np.zeros(shape=(data.shape[0], int(num_windows), size_window))
    filename_list = np.zeros(shape=(data.shape[0], int(num_windows), size_window))
    # Generate corresponding labels
    length_of_label_window = int(size_window / frame_rate * label_rate)
    step_of_label_window = int(length_of_label_window * (step_for_window / size_window))

    # Note: The sample rate for the breathing signal is different from the sample rate for the audio signal,
    # hence the timesteps are necessary to convert the labels into a single prediction later on
    if not annotated:
        windowed_labels_timesteps = np.zeros(shape=(windowed_data.shape[0], int(num_windows), length_of_label_window))
    else:
        windowed_labels = np.zeros(
            shape=(np.unique(labels['filename']).shape[0], int(num_windows), length_of_label_window))
        windowed_labels_timesteps = np.zeros(shape=windowed_labels.shape)

    all_filenames = []
    for instance_idx in range(data.shape[0]):
        sample_filenames = []
        start_idx_data = 0
        start_idx_label = 0
        temp_labels = labels[labels['filename'] == filename_dict[instance_idx]]
        temp_labels = temp_labels.drop(columns=['filename'])
        temp_labels = temp_labels.values
        for windows_idx in range(num_windows - 1):
            windowed_data[instance_idx, windows_idx] = data[instance_idx, start_idx_data:start_idx_data + size_window]
            windowed_labels_timesteps[instance_idx, windows_idx] = temp_labels[
                                                                   start_idx_label:start_idx_label + length_of_label_window,
                                                                   0]

            if annotated:
                windowed_labels[instance_idx, windows_idx] = temp_labels[
                                                             start_idx_label:start_idx_label + length_of_label_window,
                                                             1]

            start_idx_data += step_for_window
            start_idx_label += step_of_label_window
            #the timesteps have the same sampling rate as the breathing belt
            sample_filenames.append([filename_dict[instance_idx]] * length_of_label_window)
        
        if start_idx_data + size_window >= data.shape[1]:  # if the last window, if the edge of the window surpasses the
            # length of the data
            windowed_data[instance_idx, num_windows - 1] = data[instance_idx, data.shape[1] - size_window:data.shape[1]]
            windowed_labels_timesteps[instance_idx, num_windows - 1] = temp_labels[
                                                                       temp_labels.shape[0] - length_of_label_window:
                                                                       temp_labels.shape[0], 0]

            if annotated:
                windowed_labels[instance_idx, num_windows - 1] = temp_labels[
                                                                 temp_labels.shape[0] - length_of_label_window:
                                                                 temp_labels.shape[
                                                                     0], 1]
        else:
            windowed_data[instance_idx, num_windows - 1] = data[instance_idx,
                                                           start_idx_data:start_idx_data + size_window]
            windowed_labels_timesteps[instance_idx, num_windows - 1] = temp_labels[
                                                                       start_idx_label:start_idx_label + length_of_label_window,
                                                                       0]
            if annotated:
                windowed_labels[instance_idx, num_windows - 1] = temp_labels[
                                                                 start_idx_label:start_idx_label + length_of_label_window,
                                                                 1]
            start_idx_data += step_for_window
            start_idx_label += step_of_label_window
        sample_filenames.append([filename_dict[instance_idx]] * length_of_label_window)
        all_filenames.append(sample_filenames)
    return windowed_data, windowed_labels, windowed_labels_timesteps, all_filenames

class UCL_SBM_Dataset(Dataset):
    """
    A custom Dataset class for the UCL_SBM dataset.

    Attributes:

    :param data: numpy array with loaded data
    :param labels: dataframe with loaded labels
    :param timesteps: corresponding timesteps
    :param filename_list: list with corresponding filenames
    :param exp_parameters: (dict) Experimental parameters; keys vary depending on the model type.

    """
    def __init__(self, data, labels, timesteps, filename_list, exp_parameters):
        
        self.save_dir = f"embeddings/{exp_parameters['wav_model'].__class__.__name__}_w_{exp_parameters['window_size']}_s_{exp_parameters['step_size']}"
        self.data = data
        self.labels = labels
        self.timesteps = timesteps
        self.filename_list = np.vstack(filename_list)
        self.processor = exp_parameters['processor']
        self.exp_parameters = exp_parameters
        if 'speaker_embeddings' in exp_parameters:
            if exp_parameters['speaker_embeddings'] == 'wespeaker':
                self.emb_speaker = pd.read_csv(f'{OUTPUT_FOLDER}/wespeaker_features.csv', index_col = ['trials','speaker','sample_id'])
            if exp_parameters['speaker_embeddings'] == 'egemaps':
                self.emb_speaker = pd.read_csv(f'{OUTPUT_FOLDER}/opensmile_features_egemaps.csv', index_col = ['trials','speaker','sample_id'])
            if exp_parameters['speaker_embeddings'] == 'compare2016':
                self.emb_speaker = pd.read_csv(f'{OUTPUT_FOLDER}/opensmile_features_compare2016.csv', index_col = ['trials','speaker','sample_id'])
            
            self.has_speaker_emb = True
        else:
            self.has_speaker_emb = False
            self.emb_speaker = None
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio = torch.from_numpy(self.data[idx]).float()
        label = torch.from_numpy(self.labels[idx]).float()
        # Load audio embeddings, extracted beforehand. This is more efficient than calculating the embeddings for every batch
        if self.exp_parameters['use_audio_embeddings']:
            sample_id= Path(self.filename_list[idx][0]).stem       
            win_id = self.timesteps[idx][0]                   
            filename = f"{sample_id}_t{win_id}.pt"
            audio_emb = torch.load(f'{self.save_dir}/{filename}')   

        else:
            audio_emb = []
    
        if len(self.filename_list) > 1:
            filename_list = self.filename_list[idx][0]
        else:
            #idx is not relevant when working with a single file prediction
            filename_list = self.filename_list[0][0]

        if self.has_speaker_emb:
            if 'channel' in filename_list:
                match = re.search(r'speaker(\d+).*trial(\d+)', filename_list)
                speaker = int(match.group(1))
                trial = int(match.group(2))
                speaker_vector = self.emb_speaker[(self.emb_speaker.index.get_level_values('speaker') == speaker) & (self.emb_speaker.index.get_level_values('trials') == trial)] # trial 10 is the one with spontaneous speech used in the challenge
            else:    
                speaker_vector = self.emb_speaker[(self.emb_speaker.index.get_level_values('sample_id') == filename_list) & (self.emb_speaker.index.get_level_values('trials') == 10)] # trial 10 is the one with spontaneous speech used in the challenge
            
            
            speaker_vector = np.vstack(speaker_vector.to_numpy())
            speaker_vector = torch.from_numpy(speaker_vector)
        else:
            speaker_vector = None

        return audio, audio_emb, label, speaker_vector, filename_list

    def collate_fn(self, batch):
        """
        A custom collate function to process a batch of data.

        :param batch: A list of tuples containing audio and label tensors.

        :return proc_audios: A list containing the processed audio data.
        :return audios_emb: A list of the audio embeddings, extracted before the model training. Althought it leads to a slightly worse
        performance, extracting the transformer-embeddings beforehand makes the training process a lot faster
        : return labels: A list containing the labels.
        : return speaker_vector: A list containing the speaker-specific embeddings (eGeMAPS, Compare2016 or Wespeaker)
        : return filenames: A list with corresponding filenames
        
        """
        # Separate audio and labels
        audios, audios_emb, labels, speaker_vector, filenames = zip(*batch)
        
        if self.exp_parameters['use_audio_embeddings']:
            audios_emb = torch.stack(audios_emb)
        
        labels = torch.stack(labels)
        audios = torch.stack(audios)
        if speaker_vector[0] != None:
            speaker_vector = torch.stack(speaker_vector)

        # Process with Wav2Vec2 processor
        
        proc_audios = self.processor(
            audios.numpy(), 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )    
        return proc_audios, audios_emb, labels, speaker_vector, filenames