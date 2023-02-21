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
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


parser = argparse.ArgumentParser()
parser.add_argument('--audio_path', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
args = parser.parse_args()

# def test(spectrograms, model, text_transform):
#     model.eval()
#     with torch.no_grad():
#         spectrograms = torch.unsqueeze(spectrograms, dim=0)
#         spectrograms = torch.unsqueeze(spectrograms, dim=0)
#         spectrograms = spectrograms.transpose(2, 3)
#         output = model(spectrograms)
#         output = output.squeeze(0)
#         output = output.cpu().numpy()

#         decoded_preds = utils.beam_search_decoder(output, beam_width=20, n_best=20)
        
#         # Add blank symbol to the index map
#         index_map = {i-1: text_transform.index_map[i] for i in range(1, len(text_transform.index_map))}
#         index_map[len(text_transform.index_map)] = '<blank>'

#         # Decode the predictions
#         for i, step_results in enumerate(decoded_preds):
#             #print('step result:', step_results)
#             print(f"Time step {i}:")
            
#             if isinstance(step_results, int):
#                 phoneme = index_map[step_results]
#                 print(f"1-best Phoneme: {phoneme}")
                
#             else:
#                 for rank, (token, score) in enumerate(step_results):
#                     phoneme = index_map[token]
#                     print(f"{rank+1}-best Phoneme: {phoneme}, Score: {score}")
        
#     return decoded_preds


def test(spectrograms, model, text_transform, output_file):
    model.eval()
    with torch.no_grad():
        spectrograms = torch.unsqueeze(spectrograms, dim=0)
        spectrograms = torch.unsqueeze(spectrograms, dim=0)
        spectrograms = spectrograms.transpose(2, 3)
        output = model(spectrograms)
        output = output.squeeze(0)
        output = output.cpu().numpy()
        print('Output shape:', output.shape)
        decoded_preds = utils.beam_search_decoder(output, beam_width=20, n_best=5)
        
        # Add blank symbol to the index map
        index_map = {i-1: text_transform.index_map[i] for i in range(1, len(text_transform.index_map))}
        index_map[len(text_transform.index_map)] = '<blank>'

        # Append the predictions to the output file
        with open(output_file, 'a') as f:
            for i, step_results in enumerate(decoded_preds):
                print(f"Time step {i}:")
                f.write(f"Time step {i}:\n")
                
                if isinstance(step_results, int):
                    phoneme = index_map[step_results]
                    print(f"1-best Phoneme: {phoneme}")
                    f.write(f"1-best Phoneme: {phoneme}\n")
                    
                else:
                    for rank, (token, score) in enumerate(step_results):
                        phoneme = index_map[token]
                        print(f"{rank+1}-best Phoneme: {phoneme}, Score: {score}")
                        f.write(f"{rank+1}-best Phoneme: {phoneme}, Score: {score}\n")
                        
        
    return decoded_preds



def main(audio_spec, text_transform, model_path):
    hparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 512,
        "n_class": 51,
        "n_feats": 80,
        "stride":2,
        "dropout": 0.1,
    }

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda:1" if use_cuda else "cpu")
    model = asr_model.SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    audio_spec = audio_spec.to(device)
    output_file = '/home/elias/Reve/beam_inference/phoneme.txt'
    phone_pred = test(audio_spec, model, text_transform, output_file)
    return phone_pred

if __name__ == "__main__":
    audio_path = args.audio_path
    model_path = args.model_path
    waveform, sample_rate = torchaudio.load(audio_path)
    test_audio_transforms = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, n_mels=80)
    spectrogram = test_audio_transforms(waveform).squeeze(0).transpose(0, 1)
    text_transform = TextTransform()
    result = main(spectrogram, text_transform, model_path)
    #print(result)


