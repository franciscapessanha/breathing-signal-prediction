import torch.nn as nn
from transformers import AutoModel
import torch
import torch
from transformers import BatchFeature


#####################################################################
#               Hubert-based model inspired by Harma et al. (2023)
#####################################################################
class Hubert3LSTM(nn.Module):
    def __init__(self, output_size, window_size):
        super(Hubert3LSTM, self).__init__()
        self.wav_model = AutoModel.from_pretrained('facebook/hubert-large-ls960-ft')
        # Freeze the parameters
        for param in self.wav_model.parameters():
            param.requires_grad = False
        self.input_features = self.wav_model.config.hidden_size
        self.lstm = nn.LSTM(input_size=self.input_features,
                            hidden_size=64,
                            num_layers=3,
                            batch_first=True)

        self.flatten = nn.Flatten()
        if window_size == 30:
            self.fc = nn.Linear(64 * 1499, output_size)

        elif window_size == 5.12:
            self.fc = nn.Linear(64 * 255, output_size)

    def forward(self, input_values,  audio_emb, speaker_embs = [None]):
        with torch.no_grad():
            wav_output = self.wav_model(**input_values).last_hidden_state
        lstm_out, _ = self.lstm(wav_output)
        flatten_lstm = self.flatten(lstm_out)
        output = self.fc(flatten_lstm)

        return output

#####################################################################
#               Wav2Vec2-based model inspired by Mitra et al. (2024)
#####################################################################
class Wav2Vec2ConvLSTMModel(nn.Module):
    def __init__(self, output_size, window_size):
        super(Wav2Vec2ConvLSTMModel, self).__init__()
        self.wav_model = AutoModel.from_pretrained('facebook/wav2vec2-base')
        self.wav_model.encoder.layers = self.wav_model.encoder.layers[:7]
        # Freeze the Wav2Vec2 model's parameters
        for param in self.wav_model.parameters():
            param.requires_grad = False

        self.input_features = self.wav_model.config.hidden_size
        self.conv = nn.Conv1d(in_channels=self.input_features,
                              out_channels=self.input_features,
                              kernel_size=3,
                              padding=1, dilation=1)

        self.lstm = nn.LSTM(input_size=2 * self.input_features,
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 1499, output_size)  # if window size == 30 seconds

    def forward(self, audio, audio_emb, speaker_embs = [None]):
        with torch.no_grad():
            wav2vec_features = self.wav_model(**audio).last_hidden_state
        x = wav2vec_features.permute(0, 2, 1)
        conv_features = self.conv(x)
        conv_features = conv_features.permute(0, 2, 1)
        concat_features = torch.concat([wav2vec_features, conv_features], dim=-1)
        lstm_out, _ = self.lstm(concat_features)
        flatten_lstm = self.flatten(lstm_out)
        output = self.linear(flatten_lstm)
        return output


#####################################################################
#               Proposed WavLM based models
#####################################################################

class TimewiseLinear(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(feature_dim))  # Learnable weight vector

    def forward(self, X):
        return torch.matmul(X, self.weight).unsqueeze(-1)


class WavLMAttentionCNN(nn.Module):
    """
    Our proposed WavLM-Attention-CNN PyTorch model that combines WavLM large feature extraction with a transformer and CNN network.

    Attributes:
    wav_model (nn.Module): The WavLM large feature extractor from HuggingFace.
    input_features (int): The number of input features from the WavLM large model (1024).
    transformer_layers (nn.TransformerEncoderLayer): Transformer encoder layer with layer normalization.
    features (int): The number of filters for the last CNN layer (512).
    cnn_down (nn.Sequential): The CNN layers for extracting local features from the WavLM features for each time dimension. with a 1499 x 512 output.
    feature_down (nn.Sequential): The linear layer for feature downsampling for each timestep from 1499 x 512 -> 1499 x 1.
    time_down (nn.Sequential): The linear layer that takes in the 1D data and transforms it to the expected output size, from 1499 -> 750.
    """

    def __init__(self, output_size, window_size):
        super(WavLMAttentionCNN, self).__init__()

        self.output = output_size

        self.wav_model = AutoModel.from_pretrained("microsoft/wavlm-large")

        self.d_model = 1024

        self.features = 512

        # Transformer Encoder with Layer Normalization and Residual Connections
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=16,
            dropout=0.2,
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=2)

        self.cnn_down = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(self.d_model, self.features, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.features),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        self.feature_down = TimewiseLinear(self.features)
        # Timewise transformation
        self.flatten = nn.Flatten()
        
        self.time_down = nn.Linear(1499, self.output)  # Mapping time dimension to output

        self.speaker_embedding_proj = nn.Sequential(nn.Flatten(),
        nn.LazyLinear(self.features), 
            nn.ReLU(), 
            nn.Linear(self.features, 2 * self.features))

    def forward(self, audio, audio_emb, speaker_embs = [None]):
        if audio_emb[0] == []:
            x = self.wav_model(**audio)[0]
        else:
            # WavLM embeddings
            x = audio_emb.squeeze() # (B, T, 1024)

       
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1)
        x = self.cnn_down(x)
        x = x.permute(0, 2, 1)

        if speaker_embs[0] != None:
            # FiLM conditioning
            speaker_embs = speaker_embs.float()
            speaker_embs = speaker_embs.permute(0,2,1).squeeze(-1)
            gamma_beta = self.speaker_embedding_proj(speaker_embs)  # (B, 2*256)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
            x = (1 + gamma) * x + beta  # apply conditioning across time

        x = self.feature_down(x)
        x = self.flatten(x)
        x = self.time_down(x)

        return x


class WavLM_CNN(nn.Module):
    """
    Our proposed WavLM-CNN PyTorch model that combines WavLM large feature extraction with a CNN network.

    Attributes:
    wav_model (nn.Module): The WavLM large feature extractor from huggingface.
    input_features (int): The number of input features from the WavLM large model (1024).
    features (int): The number of filters for the last CNN layer.
    cnn_down (nn.Sequential): The CNN layers for extracting local features from the WavLM features for each timedemention.
    feature_downsample (nn.Linear): The linear layer for feature downsampling for each timestep from 1499 x 256 -> 1499 x 1.
    flatten (nn.Flatten): The flatten layer that flattens the output of the feature_downsample layer to 1D. So from 1499 x 1 -> 1499.
    time (nn.Linear): The linear layer that takes in the 1D data and transforms it to the expected output size. from 1499 -> 750
    """


    def __init__(self, output_size, window_size, n_layers):
        super(WavLM_CNN, self).__init__()
        
        self.wav_model = AutoModel.from_pretrained("microsoft/wavlm-large")

        self.output = output_size
        self.n_layers = n_layers

        print('number_layers: ', n_layers)
        self.d_model = 1024
        self.features = 512
        
        self.time_downsample = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Conv1d(self.d_model, self.features, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.features),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        
        self.feature_down = TimewiseLinear(self.features)
        # Timewise transformation
        self.flatten = nn.Flatten()
        self.time_down = nn.Linear(1499, self.output)  # Mapping time dimension to output

        self.unfreeze_last_n_blocks(self.n_layers)
        self.speaker_embedding_proj = nn.Sequential(nn.Flatten(),
        nn.LazyLinear(self.features), 
            nn.ReLU(), 
            nn.Linear(self.features, 2 * self.features))

    ## unfreeze last n layers from the WavLM model
    def unfreeze_last_n_blocks(self, num_blocks: int):
        for param in self.wav_model.parameters():
            param.requires_grad = False

        for i in range(0, num_blocks):
            for param in self.wav_model.encoder.layers[-1 * (i + 1)].parameters():
                param.requires_grad = True
    def forward(self, audio, audio_emb, speaker_embs = [None]):

        if self.n_layers > 0 or audio_emb[0] == []:
            x = self.wav_model(**audio)[0]
        else:
            # WavLM embeddings
            x = audio_emb.squeeze() # (B, T, 1024)
        
        x = x.permute(0, 2, 1)  
        # CNN encoder
        x = self.time_downsample(x) 
        x = x.permute(0, 2, 1)

        if speaker_embs[0] != None:
            # FiLM conditioning
            speaker_embs = speaker_embs.float()
            speaker_embs = speaker_embs.permute(0,2,1).squeeze(-1)
            gamma_beta = self.speaker_embedding_proj(speaker_embs)  # (B, 2*256)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
            x = (1 + gamma) * x + beta  # apply conditioning across time

        x = self.feature_down(x)
        x = self.flatten(x)
        x = self.time_down(x)
        return x
