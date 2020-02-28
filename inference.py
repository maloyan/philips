import numpy as np
from pathlib import Path
from tqdm import tqdm

import fastai
from fastai.metrics import accuracy
from fastai.vision import (
    models, ImageList, imagenet_stats, partial, cnn_learner, ClassificationInterpretation, to_np,
)

import os
import cv2
from matplotlib import pyplot as plt
from skimage.color import label2rgb

import albumentations as A
import random
import torch

EPOCHS        = 10
LEARNING_RATE = 1e-4
IM_SIZE       = 300

BATCH_SIZE    = 16
ARCHITECTURE  = models.resnet50

path = Path('./data')

data = (
    ImageList.from_folder(path)
    .split_by_rand_pct(valid_pct=0.2, seed=10)
    .label_from_folder()
    .transform(size=IM_SIZE)
    .databunch(bs=BATCH_SIZE)
    .normalize(imagenet_stats)
)

learn = cnn_learner(
    data,
    ARCHITECTURE,
    metrics=[accuracy],
)

learn.path = Path('.')
learn.load('model')

for i in os.listdir('./inference/'):
    img = fastai.vision.open_image(f'./inference/{i}') 
    print(i.split('.')[0], '-', str(learn.predict(img)[0]))