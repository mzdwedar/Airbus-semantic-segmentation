# Airbus-semantic-segmentation-pytorch

## Install requirements

```
pip3 install -r requirements.txt
```

### Note

*kaggle credentials **(kaggle.json)** needs to placed in path provided.*

## Train

```
python3 train.py
```

available arguments:

- `--num_epochs`: Number of epochs of training. Default=10.
- `--path`: Data path to download the dataset from kaggle. Default='./'
- `--batch_size`: Batch Size. Default=32

## Dataset

[Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection/data)
