import torchaudio
import torch
import torch.nn.functional as F
import os
import torch
import torch.utils.data as data
import torch.optim as optim
import asr_model
import mlflow
import mlflow.pytorch
import traceback
import glob
import torch.utils.data as data 



from dataset import BanglaData
from platform import python_branch
from data_utils import TextTransform
from cmath import nan
from torch import nn


os.environ['MLFLOW_TRACKING_USERNAME'] = "mlflow"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "1234567"
torch.autograd.set_detect_anomaly(True)

phone_cache = {} ##used to store phone of dataset labels computed during epoch 1 so that it does not need to be computed again


# def save_ckp(state, checkpoint_dir = './saved_model_positional_phoneme'):
#     f_path = checkpoint_dir + '/best_model_checkpoint.pt'
#     torch.save(state, f_path)

def save_ckp(state, epoch, best_loss, checkpoint_dir = './saved_model'):
    f_path = f'{checkpoint_dir}/model_checkpoint_epoch:{epoch}_val_loss:{best_loss}.pt'
    torch.save(state, f_path)

def data_processing(train_audio_transforms, valid_audio_transforms,text_transform, data, data_type="train"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    utterances = []
    for (waveform, _, utterance) in data:
        if data[0] == None or data[1] == None or data[2] == None:
            print('ohnooo')
            continue
        else:
            try:
                if data_type == 'train':
                    # print(waveform.shape)
                    spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
                else:
                    spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
            except:
                print('hayhayhay')
                continue
            # print(spec.shape)
            spectrograms.append(spec)
            # print(utterance)
            label = torch.Tensor(text_transform.text_to_int(utterance))
            labels.append(label)
            input_lengths.append(spec.shape[0]//2)
            label_lengths.append(len(label))
            utterances.append(utterance)
    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    return spectrograms, labels, utterances, input_lengths, label_lengths

def train(model, device, train_loader, test_loader, criterion, optimizer, scheduler, epoch):
    model.train()
    data_len = len(train_loader.dataset)
    count_train = 0
    tot_train_loss = []
    for batch_idx, _data in enumerate(train_loader):
        spectrograms, labels,utterances, input_lengths, label_lengths = _data 
        spectrograms, labels = spectrograms.to(device), labels.to(device)

        optimizer.zero_grad()

        output = model(spectrograms)  # (batch, time, n_class)
        output = F.log_softmax(output, dim=2)


        output = output.transpose(0, 1) # (time, batch, n_class)
        loss = criterion(output, labels, input_lengths, label_lengths)
        tot_train_loss.append(loss.item())
        if loss == nan:
            print('loss for this batch is NaN')
        loss.backward()


        optimizer.step()
        scheduler.step()
        count_train += 1
        if batch_idx % 100 == 0 or batch_idx == data_len:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(spectrograms), data_len,
                100. * batch_idx / len(train_loader), loss.item()))
    
    model.eval()
    tot_val_loss = []
    count_val = 0
    with torch.no_grad():
        for batch_idx, _data in enumerate(test_loader):
            spectrograms, labels, utterances, input_lengths, label_lengths = _data 
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1) # (time, batch, n_class)

            loss = criterion(output, labels, input_lengths, label_lengths)
            count_val += 1
            tot_val_loss.append(loss.item())
    
    mlflow.log_metric("Average Train Loss", float(sum(tot_train_loss)/count_train), step=epoch)
    mlflow.log_metric("Average Validation Loss", float(sum(tot_val_loss)/count_val), step=epoch)
           
    return float(sum(tot_val_loss)/count_val), model, optimizer
    


def main(train_audio_transforms, valid_audio_transforms, text_transform, learning_rate=5e-4, batch_size=20, epochs=10):
    
    # hparams = {
    #     "n_cnn_layers": 3,
    #     "n_rnn_layers": 5,
    #     "rnn_dim": 512,
    #     "n_class": 148,
    #     "n_feats": 80,
    #     "stride":2,
    #     "dropout": 0.1,
    #     "learning_rate": learning_rate,
    #     "batch_size": batch_size,
    #     "epochs": epochs
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

    # model config for training with larger dataset(all_data) 
    # hparams = {
    #     "n_cnn_layers": 5,
    #     "n_rnn_layers": 7,
    #     "rnn_dim": 512,
    #     "n_class": 51,
    #     "n_feats": 80,
    #     "stride":2,
    #     "dropout": 0.1,
    #     "learning_rate": learning_rate,
    #     "batch_size": batch_size,
    #     "epochs": epochs
    # }

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    

    train_dataset = BanglaData(data_folder = './trainable_dataset/train')
    test_dataset = BanglaData(data_folder = './trainable_dataset/valid')
    

    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=True,
                                collate_fn=lambda x: data_processing(train_audio_transforms, valid_audio_transforms, text_transform, x, 'train'))
    test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=False,
                                collate_fn=lambda x: data_processing(train_audio_transforms, valid_audio_transforms, text_transform, x, 'valid'))
    # for __data in train_loader:
    #     print(__data)
    model = asr_model.SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
        ).to(device)
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7]) ## for 8 GPUs
    model = model.to(device)
    # print(model)
    # print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    optimizer = optim.AdamW(model.parameters(), hparams['learning_rate'])
    criterion = nn.CTCLoss(blank=0, zero_infinity=True).to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hparams['learning_rate'], 
                                            steps_per_epoch=int(len(train_loader)),
                                            epochs=hparams['epochs'],
                                            anneal_strategy='linear')

    training_parameters = {
        "Batch Size": hparams['batch_size'],
        "Epochs": epochs,
        # "Optimizer": optimizer,
        "Learning Rate": hparams['learning_rate'],
        "Number of CNN layers used": hparams['n_cnn_layers'],
        "Number of RNN layers used": hparams['n_rnn_layers']
    }


    best_loss = 100000000000000000
    path = './saved_model'
    # save_path = os.path.join(path,'phoneme_prediction_model.pt')
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    

    with mlflow.start_run(experiment_id=experiment_id):
        mlflow.log_params(training_parameters)
        for epoch in range(1, epochs + 1):
            temp_loss, model, optimizer = train(model, device, train_loader, test_loader, criterion, optimizer, scheduler, epoch)
            if temp_loss < best_loss:
                checkpoint = {
                            'epoch': epoch,
                            'state_dict': model.module.state_dict(),
                            'optimizer': optimizer.state_dict()
                            }
                print(f'validation loss decreased from {best_loss} to {temp_loss}, model being saved')
                best_loss = temp_loss
                save_path = f'{path}/phoneme_prediction_model_{epoch}_val_loss:{best_loss}.pt'
                # torch.save(model.state_dict(), save_path)
                torch.save(model.module.state_dict(), save_path)
                save_ckp(checkpoint, epoch, best_loss)
      
      

        


if __name__ == "__main__":

    train_audio_transforms = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, n_mels=80)
        # torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
        # torchaudio.transforms.TimeMasking(time_mask_param=35)
    )
    
    valid_audio_transforms = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, n_mels=80)
    # valid_audio_transforms = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80),

    text_transform = TextTransform()
    
    learning_rate = 2e-5
    batch_size = 350
    epochs = 100
    

    
 
    mlflow.set_tracking_uri("http://119.148.4.20:6060/")
    experiment_id = mlflow.get_experiment_by_name("stt_deepSpeech2 positional phoneme (clean data)")
    

    if experiment_id is None:
        experiment_id = mlflow.create_experiment("stt_deepSpeech2 positional phoneme (clean data)")
    else:
        experiment_id = experiment_id.experiment_id
    


    main(train_audio_transforms, valid_audio_transforms, text_transform, learning_rate, batch_size, epochs)
    
