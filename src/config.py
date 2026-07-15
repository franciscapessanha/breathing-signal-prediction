import torch

from model_definition import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to use across the experiments
# ===================================
MODELS_FOLDER = 'data/models/'
OUTPUT_FOLDER = 'data/output/'
AUDIO_INTERSPEECH = 'data/wav/'
BREATH_INTERSPEECH = 'data/ground truth/'
UCL_DATA = 'data/full UCL dataset/'

# Experimental parameters for models proposed in %Add paper reference after publication%
# ======================================================================================

# HUBERT
# ======
HUBERT_REPLICATED = {'model_name': 'Hubert-3LSTM',
                     'model_class': Hubert3LSTM,
                     'window_size': 5.12,
                     'step_size': 5.12 * 2 / 5,
                     'batch_size': 128,
                     'epochs': 128,
                     'loss': 'PCC',
                     'optimizer': 'SGD',
                     'learning_rate': 0.01,
                     'processor': AutoProcessor.from_pretrained('facebook/hubert-large-ls960-ft'),
                     'wav_model': AutoModel.from_pretrained('facebook/hubert-large-ls960-ft'),
                     'use_audio_embeddings':  False}

HUBERT_EXTENDED = {'model_name': 'Hubert-3LSTM',
                   'model_class': Hubert3LSTM,
                   'window_size': 30,
                   'step_size': 6,
                   'batch_size': 64,
                   'epochs': 128,
                   'loss': 'PCC',
                   'optimizer': 'Adam',
                   'learning_rate': 0.001,
                   'processor': AutoProcessor.from_pretrained('facebook/hubert-large-ls960-ft'),
                   'wav_model': AutoModel.from_pretrained('facebook/hubert-large-ls960-ft'),
                   'use_audio_embeddings':  False}


# Wav2Vec2
# ========
WAV2VEC2_REPLICATED = {'model_name': 'Wav2vec2-LSTM',
                       'model_class': Wav2Vec2ConvLSTMModel,
                       'window_size': 30,
                       'step_size': 30,
                       'batch_size': 64,
                       'epochs': 128,
                       'loss': 'PCC',
                       'optimizer': 'Adam',
                       'learning_rate': 0.005,
                       'processor': AutoProcessor.from_pretrained('facebook/wav2vec2-base'),
                       'wav_model': AutoModel.from_pretrained('facebook/wav2vec2-base'),
                       'use_audio_embeddings':  False}

WAV2VEC2_EXTENDED = {'model_name': 'Wav2vec2-LSTM',
                     'model_class': Wav2Vec2ConvLSTMModel,
                     'window_size': 30,
                     'step_size': 6,
                     'batch_size': 64,
                     'epochs': 128,
                     'loss': 'PCC',
                     'optimizer': 'Adam',
                     'learning_rate': 0.005,
                     'processor': AutoProcessor.from_pretrained('facebook/wav2vec2-base'),
                     'wav_model': AutoModel.from_pretrained('facebook/wav2vec2-base'),
                     'use_audio_embeddings':  False}


# Our proposed models
# ====================

# WAVLM ATTENTION
# # ====================

BASE_WAVLM_ATTENTION = {
    'model_class': WavLMAttentionCNN,
    'window_size': 30,
    'step_size': 6,
    'batch_size': 10,
    'epochs': 128,
    'optimizer': 'AdamW',
    'learning_rate': 2e-4,
    'patience': 7,
    'processor': Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large"),
    'wav_model': AutoModel.from_pretrained("microsoft/wavlm-large")
}

WAVLM_ATTENTION = {
    **BASE_WAVLM_ATTENTION,
    'model_name': 'WavLM-Attention-CNN',
    'loss': 'PCC',
    'use_audio_embeddings': False
}

WAVLM_ATTENTION_NO_EMB = {
    **BASE_WAVLM_ATTENTION,
    'model_name': 'WavLM-Attention-CNN-no-emb',
    'loss': 'PCC',
    'use_audio_embeddings':  False
}
WAVLM_ATTENTION_OLD = {
    **BASE_WAVLM_ATTENTION,
    'model_name': 'WavLM-Attention-CNN_old',
    'loss': 'PCC',
    'use_audio_embeddings':  False
}
WAVLM_ATTENTION_EGEMAPS = {
    **BASE_WAVLM_ATTENTION,
    'model_name': 'WavLM-Attention-CNN-egemaps',
    'loss': 'PCC',
    'use_audio_embeddings':  False,  'speaker_embeddings': 'egemaps'}

WAVLM_ATTENTION_WESPEAKER = {
    **BASE_WAVLM_ATTENTION,
    'model_name': 'WavLM-Attention-CNN-wespeaker',
    'loss': 'PCC',
    'use_audio_embeddings':  False,  'speaker_embeddings': 'wespeaker'}

# WavLM-CNN Models
#=============================================

BASE_WAVLM_CNN = {
    'window_size': 30,
    'step_size': 6,
    'batch_size': 8,
    'loss': 'PCC',
    'optimizer': 'AdamW',
    'learning_rate': 2e-4,
    'processor': Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large"),
    'wav_model': AutoModel.from_pretrained("microsoft/wavlm-large"),
    'patience': 7
    } 


def make_wavlm_cnn_config(
    n_layers: int,
    model_class,
    model_name: str,
    epochs: int = 128, use_audio_embeddings = False
):
    return {
        **BASE_WAVLM_CNN,
        'model_name': model_name,
        'model_class': model_class,
        'n_layers': n_layers,
        'epochs': epochs,
        'use_audio_embeddings': use_audio_embeddings
    }

WAVLM_CNN_16 = make_wavlm_cnn_config(
    n_layers=16,
    model_class=WavLM_CNN,
    model_name='WavLMCNN-16ft')

WAVLM_CNN_12 = make_wavlm_cnn_config(
    n_layers=12,
    model_class=WavLM_CNN,
    model_name='WavLMCNN-12ft'
)

WAVLM_CNN_8 = make_wavlm_cnn_config(
    n_layers=8,
    model_class=WavLM_CNN,
    model_name='WavLMCNN-8ft'
)

WAVLM_CNN_4 = make_wavlm_cnn_config(
    n_layers=4,
    model_class=WavLM_CNN,
    model_name='WavLMCNN-4ft'
)

WAVLM_CNN_0 = make_wavlm_cnn_config(
    n_layers=0,
    model_class=WavLM_CNN,
    model_name='WavLMCNN-0ft'
)

# Speaker Embeddings (Base model)
#=============================================
WAVLM_CNN_WESPEAKER = {**make_wavlm_cnn_config(
    n_layers=0,
    model_class=WavLM_CNN,
    model_name='WavLMCNN-0ft-wespeaker'), 'speaker_embeddings': 'wespeaker'}

WAVLM_CNN_EGEMAPS = {**make_wavlm_cnn_config(
    n_layers=0,
    model_class=WavLM_CNN,
    model_name='WavLMCNN-0ft-egemaps'), 'speaker_embeddings': 'egemaps'}

WAVLM_CNN_COMPARE = {**make_wavlm_cnn_config(
    n_layers=0,
    model_class=WavLM_CNN,
    model_name='WavLMCNN-0ft-compare2016'), 'speaker_embeddings': 'compare2016'}


# Speaker Embeddings and 16 finetuned layers
#=============================================
WAVLM_CNN_WESPEAKER_16 = {**make_wavlm_cnn_config(
    n_layers=16,
    model_class=WavLM_CNN,
    model_name='WavLMCNN-16ft-wespeaker'), 'speaker_embeddings': 'wespeaker'}

WAVLM_CNN_EGEMAPS_16 = {**make_wavlm_cnn_config(
    n_layers=16,
    model_class=WavLM_CNN,
    model_name='WavLMCNN-16ft-egemaps'), 'speaker_embeddings': 'egemaps'}

WAVLM_CNN_COMPARE_16 = {**make_wavlm_cnn_config(
    n_layers=16,
    model_class=WavLM_CNN,
    model_name='WavLMCNN-16ft-compare2016'), 'speaker_embeddings': 'compare2016'}

# Correspondence between UCL-SBM dataset and the subset used for the Compare2020 Interspeech Challenge
MAPPING = {
    "devel_00.wav": "speaker33",
    "devel_01.wav": "speaker37",
    "devel_02.wav": "speaker38",
    "devel_03.wav": "speaker39",
    "devel_04.wav": "speaker40",
    "devel_05.wav": "speaker43",
    "devel_06.wav": "speaker44",
    "devel_07.wav": "speaker45",
    "devel_08.wav": "speaker46",
    "devel_09.wav": "speaker47",
    "devel_10.wav": "speaker48",
    "devel_11.wav": "speaker49",
    "devel_12.wav": "speaker50",
    "devel_13.wav": "speaker51",
    "devel_14.wav": "speaker53",
    "devel_15.wav": "speaker54",
    "test_00.wav": "speaker12",
    "test_01.wav": "speaker15",
    "test_02.wav": "speaker16",
    "test_03.wav": "speaker17",
    "test_04.wav": "speaker24",
    "test_05.wav": "speaker25",
    "test_06.wav": "speaker26",
    "test_07.wav": "speaker27",
    "test_08.wav": "speaker28",
    "test_09.wav": "speaker29",
    "test_10.wav": "speaker30",
    "test_11.wav": "speaker31",
    "test_12.wav": "speaker32",
    "test_13.wav": "speaker34",
    "test_14.wav": "speaker35",
    "test_15.wav": "speaker36",
    "train_00.wav": "speaker1",
    "train_01.wav": "speaker2",
    "train_02.wav": "speaker3",
    "train_03.wav": "speaker4",
    "train_04.wav": "speaker5",
    "train_05.wav": "speaker6",
    "train_06.wav": "speaker7",
    "train_07.wav": "speaker8",
    "train_08.wav": "speaker9",
    "train_09.wav": "speaker10",
    "train_10.wav": "speaker11",
    "train_11.wav": "speaker13",
    "train_12.wav": "speaker14",
    "train_13.wav": "speaker18",
    "train_14.wav": "speaker19",
    "train_15.wav": "speaker21",
    "train_16.wav": "speaker22",
}