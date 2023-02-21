import torchaudio
import torch
import utils
import torch.nn.functional as F
import torch.utils.data as data
import asr_model
import argparse
import math
import numpy as np
import tensorflow as tf
from torch import nn
from dataset import BanglaData
from data_utils import TextTransform

parser = argparse.ArgumentParser()
parser.add_argument('--audio_path', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
args = parser.parse_args()

def test(spectrograms, model, text_transform):
    #print('inference on going')
    model.eval()
    with torch.no_grad():
        spectrograms = torch.unsqueeze(spectrograms, dim=0)
        spectrograms = torch.unsqueeze(spectrograms, dim=0)
        spectrograms = spectrograms.transpose(2, 3)
        output = model(spectrograms)
        output = F.log_softmax(output, dim=2)
        #print(output) 
        
        output = output.squeeze(0)
        # print(output.shape) 
        #print('Output shape:',output.shape) 
        output = output.cpu()
        output = np.array(output)
   
        decoded_preds = utils.BeamSearch(output)
        #decoded_pred_lm= utils.beamsearch_LM(output)
       
      
    return decoded_preds
        
       

def main(audio_spec, text_transform, model_path):
    
    # hparams = {
    #     "n_cnn_layers": 3,
    #     "n_rnn_layers": 5,
    #     "rnn_dim": 512,
    #     "n_class": 51,
    #     "n_feats": 80,
    #     "stride":2,
    #     "dropout": 0.1
    # }

    hparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 512,
        "n_class": 51,
        "n_feats": 80,
        "stride":2,
        "dropout": 0.1,
    }

    # hparams = {
    #     "n_cnn_layers": 5,
    #     "n_rnn_layers": 7,
    #     "rnn_dim": 512,
    #     "n_class": 51,
    #     "n_feats": 80,
    #     "stride":2,
    #     "dropout": 0.1
    # }
    
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda:1" if use_cuda else "cpu")
    model = asr_model.SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
        ).to(device)
    
   
    model.load_state_dict(torch.load(model_path))
    audio_spec = audio_spec.to(device)
    phone_pred = test(audio_spec,model,text_transform)
    return phone_pred
    
if __name__ == "__main__":
    # audio_path = '/home/fahim/deepspeech2/b89a87ab-57cf-4281-bd5b-89779095f169.flac' ##provide path of the audio file you want to perform inference on
    audio_path = args.audio_path
    model_path = args.model_path
    waveform, sample_rate = torchaudio.load(audio_path)
    test_audio_transforms = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, n_mels=80)
    spectogram = test_audio_transforms(waveform).squeeze(0).transpose(0, 1)
    #print('Spectogram shape:',spectogram.shape)
    text_transform = TextTransform()

    result = main(spectogram,text_transform, model_path)
    print(result)

