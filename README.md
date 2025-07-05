# DeepSpeech2 – Automatic Speech Recognition (ASR) Model

This repository contains an implementation of the **DeepSpeech2** model for end-to-end automatic speech recognition (ASR). The project demonstrates two versions (`v1` and `v2`) of the model architecture, data utilities, training pipelines, and inference logic, suitable for natural language processing and speech-to-text tasks.

---

## 📂 Project Structure

### `deepspeech_v1/`
Contains the initial version of the DeepSpeech2 implementation.

- `asr_model.py` – Core DeepSpeech2 model architecture
- `data_utils.py` – Utility functions for loading and preparing datasets
- `dataset.py` – Dataset preprocessing logic
- `inference_beam.py` – Beam search decoding during inference
- `utils.py` – General helper functions

---

### `deepspeech_v2/`
An improved or updated version of the DeepSpeech2 ASR pipeline.

- `asr_model.py` – Updated model structure
- `data_utils.py` – Updated utilities for preprocessing
- `dataset.py` – Dataset construction
- `inference.py` – Inference logic for decoding predictions
- `phoneme_prediction_model.pt` – Pretrained model checkpoint
- `train.py` – Training script to fine-tune the model
- `utils.py` – Utility functions shared across components

---

## 🚀 Getting Started

1. **Install Dependencies**

Make sure you have PyTorch, NumPy, and other required packages installed:

```bash
pip install -r requirements.txt
```

2. **Train the Model**

Navigate to `deepspeech_v2/` and run:

```bash
python train.py
```

3. **Run Inference**

To transcribe speech using the trained model:

```bash
python inference.py
```

---

## 📌 Notes

- You may need to provide your own dataset or preprocessing scripts for speech inputs.
- Ensure proper CUDA/GPU support if training large models.
- The `phoneme_prediction_model.pt` file is a placeholder pretrained model used for quick evaluation.

---

## 📧 Contact

For questions or collaboration, feel free to reach out via GitHub or email.
