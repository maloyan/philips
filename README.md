INFERENCE

1) To run this model you need to place 3 folders from google drive in the same directory as inference.py
https://drive.google.com/open?id=1JHx2CqMwRK2M3ufAZ3iRckty-w6obTV5

2) docker build . -t philips
3) nvidia-docker run -it -v "$(pwd)":/workspace/philips philips bash
4) python inference.py

It will show a name of test images from ./inference/ folder and predicted class:
lamp - wake-up-light
bottle - smart-baby-bottle
shaver - shaver

If you want to get predictions for your images you should delete test images in ./inference and replace with your


TRAINING

If you want to see the process of model training, you should check Generating dataset and training.ipynb. Basically, it creates augmented images and then trains on it using pretrained resnet50
