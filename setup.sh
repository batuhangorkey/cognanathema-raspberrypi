# !/bin/bash

sudo apt update
sudo apt upgrade -y
sudo apt install -y python3-pip
sudo apt install -y python3-picamera2 python3-opencv
sudo apt install -y python3-libcamera python3-kms++ libcap-dev
sudo apt install -y python3-prctl libatlas-base-dev ffmpeg 

pip3 install -r requirements.txt
