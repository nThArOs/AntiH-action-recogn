# Action-recognition-on-antihpert-dataset
A quick code allowing to train and run an VIDEOMAE acttion classification algorithm on antihpert dataset.

## Introduction

### Code details

Vid_compression.py

inference_finetuned_model (test)

videomae_finetune.py (train)

### Data to downloads

https://zenodo.org/records/14825645

weight (Already trained and ready to use weight)

Antihpert-gre (Anti Hpert dataset cut into small video clip, there are total 22 difference operations and 2500 videos)

--- Same dataset but the compressed representation
compressed

compressed_antihpert-gre


## Usage

### Train
```
$ python3 
```
### Test
```
$ python3 test.py 
```
### Detect
```
$ python3 detect.py 
```
### Transform
```
$ python3 ann_to_snn.py
```
For higher accuracy(mAP), you can try to adjust some hyperparameters.

*Trick: the larger timesteps, the higher accuracy.*
