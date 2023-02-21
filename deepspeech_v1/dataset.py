import os
import torch
import torchaudio
import glob
import csv
import librosa
from typing import Tuple
from torch import Tensor
from torch.utils.data import Dataset

class BanglaData(Dataset):
    
    def __init__(self, data_folder: str = './trainable_dataset/train'):
        
        self.audio_file_paths = glob.glob(data_folder+'/*.flac')
        self.text_file_paths = []
        
        for aud_path in self.audio_file_paths:
            txt_guid = aud_path.split('/')[-1].split('.')[0]
            text_path = f"{data_folder}/{txt_guid}.txt"
            text_path = text_path.replace('_DeepFilterNet2.txt','.txt')
            self.text_file_paths.append(text_path)

    
    # def load_item(self, n: int) -> Tuple[Tensor, int, str]:
    #     audio_path = self.audio_file_paths[n]
    #     text_path = self.text_file_paths[n]
    #     # Load audio
    #     try:
    #         waveform, sample_rate = librosa.load(audio_path, mono = True, sr = None)
    #         if sample_rate != 16000:
    #             waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
    #             sample_rate = 16000
    #         waveform = torch.from_numpy(waveform)
    #         utterance = ''.join(open(text_path, 'r', encoding='utf-8').readlines())
    #     except:
    #         # filename = audio_path.split('/')[-1]
    #         with open('debug.txt', 'a') as f:
    #             f.write(audio_path.split('/')[-1]+'\n')
            
    #         waveform = None
    #         sample_rate = None
    #         utterance = None
            
    #     # waveform, sample_rate = torchaudio.load(audio_path)
    #     # Load text
    #     # utterance = '। '.join(open(text_path, 'r', encoding='utf-8').readlines())
        
    #     return (
    #         waveform,
    #         sample_rate,
    #         utterance,
    #     )

    def load_item(self, n: int) -> Tuple[Tensor, int, str]:
        audio_path = self.audio_file_paths[n]
        text_path = self.text_file_paths[n]
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        # Load text
        utterance = ''.join(open(text_path, 'r', encoding='utf-8').readlines())
        return (
            waveform,
            sample_rate,
            utterance,
        )
    
    def __getitem__(self, n: int) -> Tuple[Tensor, int, str]:
        # data = self.load_item(n)
        # if data[0] == None or data[1] == None or data[2]  == None:
        #     return data
        # else:
        #     return None
        return self.load_item(n)

    def __len__(self) -> int:
        return len(self.audio_file_paths)
    
if __name__ == "__main__":
    b = BanglaData()
    
    # print(b.__len__())
    