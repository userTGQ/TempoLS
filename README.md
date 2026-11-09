# TempoLS

This repository provides the PyTorch implementation of **TempoLS**. We hope that this code will be useful for your research and applications.

## Requirements

The code is implemented and tested with the following environment:

- Python 3.9.12
- PyTorch 1.11.0
- NumPy 1.21.2

## Datasets

For the original datasets, please refer to the [STSGCN repository](https://github.com/Davidham3/STSGCN), which provides PEMS03, PEMS04, PEMS07, and PEMS08.

We sincerely thank the dataset providers for making these datasets publicly available.

## Data Preprocessing

Run `prepare_pems_calendar_splits.py` to generate time-aware training, validation, and test sets for PEMS03, PEMS04, PEMS07, and PEMS08, incorporating calendar-based time features.

```bash
python python prepare_pems_calendar_splits.py
```

## Training

After preprocessing the datasets, you can start training TempoLS.

The following example demonstrates how to train TempoLS on PEMS08:

```bash
nohup python -u train.py --data PEMS08 > PEMS08.log 2>&1 &
```

You can customize the model configuration and training hyperparameters in `train.py`.

## Reference Code

The implementation was developed based on the [LibCity](https://github.com/Aoyu-Liu/STAMT) framework, an open-source framework for traffic forecasting.
