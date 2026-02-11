# DeepVLF: Deep Variable-Length Feedback Codes
### Official PyTorch Implementation 

This repository provides a reference implementation of **Deep Variable-Length Feedback (DeepVLF) codes**, including receiver-driven, transmitter-driven, and hybrid termination strategies, as described in our paper:

> **Deep Variable-Length Feedback Codes**  
> https://arxiv.org/abs/2602.07881

---

## Repository Structure

The codebase is organized into **two main directories**, each corresponding to a specific termination paradigm:

```text
.
├── DeepVLF_R_and_hybrid/           # Receiver-terminated and hybrid DeepVLF implementations
│   ├── weights/                       # Pre-trained model checkpoints (partial)
│   ├── main.py                        # Main file
│   ├── parameter.py                   # Hyperparameter and experiment configuration settings
│   ├── model.py                       # DeepVLF-R and hybrid scheme architectures
│   ├── log.txt                        # Training log for DeepVLF-R
│   ├── log_hybrid.txt                 # Training log for hybrid scheme
│   └── ...                            
│
├── DeepVLF_T/                      # Transmitter-terminated DeepVLF implementation
│   ├── weights/                       # Pre-trained model checkpoints (partial)
│   ├── main.py                        # Main file
│   ├── parameter.py                   # Hyperparameter and experiment configuration settings
│   ├── model.py                       # DeepVLF-T architecture
│   ├── log.txt                        # Training log for DeepVLF-T
│   └── ...                            
│
├── environment.yaml                # Conda environment specification
├── LICENSE                         # License information
└── README.md                       # Project documentation    

```


## Pretrained Models and Data Availability

Due to the large number of trained models and the substantial size of the training data, this repository includes only a small subset of representative pretrained models as examples.
The complete collection of trained models and training data used in the paper is hosted on Google Drive:
> https://drive.google.com/drive/folders/1CTNEH3AJJYb2y2tx_N8dqNrLM3uY1KPf?usp=sharing

---

## Model Overview

The three DeepVLF models considered in this work are distributed across the two directories as follows:

| Model            | Termination Strategy                  | Directory                  |
|------------------|--------------------------------------|----------------------------|
| DeepVLF-R        | Receiver-driven                       | DeepVLF_R_and_hybrid/      |
| Hybrid scheme    | Joint (Receiver + Transmitter)        | DeepVLF_R_and_hybrid/      |
| DeepVLF-T        | Transmitter-driven                    | DeepVLF_T/                 |


---

## Environment Setup

The required software environment can be created using Conda:
```bash
conda env create -f environment.yaml
```

## Training

DeepVLF_R and Hybrid scheme:
```bash
python DeepVLF_R_and_hybrid/main.py \ 
    --train=1 \ 
    --snr1=2 \ 
    --snr2=100 \ 
    --fading_process=0 \ 
    --model_name='vf_new' \ 
    --lr=0.0001 \ 
    --truncated=10 \ 
    --batchSize=2048 \ 
    --totalbatch=40 \ 
    --core=4 \ 
    --restriction='high' 
```
DeepVLF_T:
```bash
python DeepVLF_T/main.py \ 
    --train=1 \ 
    --snr1=2 \ 
    --snr2=100 \ 
    --real_fading=0 \ 
    --sigma1=0 \ 
    --sigma2=0 \
    --model_name='vt_new' \ 
    --lr=0.0001 \ 
    --max_t=20 \ 
    --batchsize=2048 \ 
    --belief_threshold_tx=0.9 \ 
    --belief_threshold_rx=0.9999  \ 
    --train_steps=160000
```



## Evaluation

DeepVLF_R:
(Please make sure 'T=0' in model.py, line 32 ('termination' function).)
```bash
python DeepVLF_R_and_hybrid/main.py \ 
    --train=0 \ 
    --fading_process=0 \ 
    --snr1=1.5 \ 
    --snr2=100 \ 
    --model_name='vf_22' \ 
    --truncated=10 \ 
    --batchSize=2048 \ 
    --totalbatch=40 \ 
    --restriction='high' \ 
    --test_model='weights/vf_22/latest'
```
Hybrid scheme:
(Please make sure 'T=1' in model.py, line 32 ('termination' function).)
```bash
python DeepVLF_R_and_hybrid/main.py \ 
    --train=0 \ 
    --fading_process=0 \ 
    --snr1=1.5 \ 
    --snr2=100 \ 
    --model_name='vf_22' \ 
    --truncated=10 \ 
    --batchSize=2048 \ 
    --totalbatch=40 \ 
    --restriction='high' \ 
    --test_model='weights/vf_22/latest'
```
DeepVLF_T:
```bash
python DeepVLF_T/main.py \ 
    --train=0 \ 
    --snr1=2 \ 
    --snr2=100 \ 
    --real_fading=0 \ 
    --sigma1=0 \ 
    --sigma2=0 \
    --model_name='vt_11' \ 
    --model_weights='weights/vt_11/latest' \ 
    --max_t=20 \ 
    --batchsize=16384 \ 
    --belief_threshold_tx=0.9 \ 
    --belief_threshold_rx=0.9999 \ 
    --test_steps=50000 
```


## Citation

If you find this repository useful, please consider citing the accompanying paper:
```text
@article{DeepVLF,
  title   = {Deep Variable-Length Feedback Codes},
  author  = {Yu Ding and Yulin Shao},
  journal = {arXiv:2602.07881},
  year    = {2026}
}
```



