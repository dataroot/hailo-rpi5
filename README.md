# Hailo RPi5

This repository provides a detection-tracker algorithm for the Raspberry Pi 5, utilizing the AI capabilities of the Hailo-8L chip. The algorithm is implemented within a GStreamer pipeline, specifically designed to track the first object detected by the YOLO model.

## Overview

This project offers a detection and tracking solution integrated with the GStreamer pipeline. It aims to track objects identified by the YOLO model, enhancing the AI capabilities of your Raspberry Pi 5 setup.

## Prerequisites

Before running this code, ensure you have the following hardware and software set up:

- Raspberry Pi 5
- AI Kit, based on Hailo-8L
- Display connected to the Raspberry Pi

Find how to set them correctly [here](https://datarootlabs.com/blog/rpi-ai-kit).

## How to run?

### Initialize and Update Submodules

First, clone the repository:
```bash
git clone --recursive https://github.com/dataroot/hailo-rpi5.git
```

### Configure the environment

To set up the environment, run the following commands:
```bash
cd hailo-rpi5/hailo-rpi5-examples
source setup_env.sh
pip install -r requirements.txt
./download_resources.sh
cd ..
```
If you already have a configured environment, simply activate it:
```bash
cd hailo-rpi5/hailo-rpi5-examples
source setup_env.sh
cd ..
```

### Export display

Export display to visualize the application's output:
```bash
export DISPLAY=:0
```

### Run the application

Finally, execute the application:
```bash
python custom_pipeline.py
```

If you want to save the processed video, add the `--save` argument with the directory and filename with the `.mkv` extension:
```bash
python custom_pipeline.py --save "directory/to/file.mkv"
```

## Example of running:
The following are examples of the application running, illustrating its tracking capabilities exclusively:
![Example of tracking](example1.gif)

![Example of tracking](example3.gif)

![Example of tracking](example2.gif)

