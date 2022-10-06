from zipfile import ZipFile
import pandas as pd
import os
from subprocess import call


def extract(path):
    if ('train_v2' not in os.listdir(path)):
        call('kaggle competitions download -c airbus-ship-detection ', shell=True)


    with ZipFile('airbus-ship-detection.zip', 'r') as zipObj:
        zipObj.extract('train_ship_segmentations_v2.csv')

    segments = pd.read_csv('train_ship_segmentations_v2.csv', index_col=0).dropna().reset_index()

    segments = segments.groupby("ImageId")[['EncodedPixels']].agg(lambda rle_codes: ' '.join(rle_codes)).reset_index()

    # segments = segments[:7000]

    with ZipFile(os.path.join(path, 'airbus-ship-detection.zip'), 'r') as zipObj:
        for file in segments['ImageId'].values:
            file = os.path.join('train_v2', file)
            zipObj.extract(file)

    call('rm airbus-ship-detection.zip', shell=True)

    segments = segments.sample(frac=1, random_state=42).reset_index(drop=True)

    return segments