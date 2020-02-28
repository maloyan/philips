FROM nvcr.io/nvidia/pytorch:19.11-py3

ADD . /workspace/philips
WORKDIR /workspace/philips
RUN pip install fastai albumentations opencv-python==4.2.0.32 matplotlib numpy tqdm Pillow==6.2.0 
