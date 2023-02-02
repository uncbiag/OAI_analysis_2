#!/usr/bin/env bash

set -xeu

# VTK deps
sudo apt-get update
sudo apt-get install git -y
sudo apt-get install ffmpeg libsm6 libxext6 -y
