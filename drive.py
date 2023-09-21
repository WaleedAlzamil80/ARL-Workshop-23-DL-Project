import argparse
import base64
from datetime import datetime
import os
import shutil

from model import *

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

import utils

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

MAX_SPEED = 25
MIN_SPEED = 10

speed_limit = MAX_SPEED
image_buffer = []
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        # save frame
        
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
        try:
    
            # predict the steering angle for the image
            # steering_angle = float(model.predict(image, batch_size=1))
            # lower the throttle as the speed increases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            # Convert PIL Image to PyTorch Tensor and apply transformations if any

            transform = transforms.Compose([transforms.ToTensor()])
            image = transform(image).unsqueeze(0).to(device)
            # predict the steering angle for the image
            # steering_angle = float(model.predict(image, batch_size=1))
            global image_buffer 
            # Add the new image to the buffer
            image_buffer.append(image)

            # Now you should always have the last three images in your buffer.
            if len(image_buffer) == 3:
                images = torch.cat(image_buffer, dim=0)

                with torch.no_grad():
                    images = reshape_images(images)
                    output = model(images)   # Pass the concatenated images to the model
                steering_angle = float(output[0][0].item())
                throttle = float(output[0][1].item())

                # reset the buffer to take the new images
                image_buffer = []

                global speed_limit
                if speed > speed_limit:
                    speed_limit = MIN_SPEED  # slow down
                else:
                    speed_limit = MAX_SPEED
                # throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2 Modified
                throttle = float(output[0][1].item()) # Modified

                print('{} {} {}'.format(steering_angle, throttle, speed))
                send_control(steering_angle, throttle)
        except Exception as e:
            print(e)
        
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model pth file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(args.model, map_location=device).to(device) # torch.save(model, filepath) the write way to save the model
    model.eval()
    # Assuming your model is already loaded as `model`
    dummy_input = torch.randn(1, 3, 3, 320, 160)
    try:

        dummy_input = reshape_images(dummy_input).to(device)

        # Pass the dummy input through the model
        output = model(dummy_input)

    except Exception as e:
        print("Check the reshape_images and make sure it returns the expected shape of your model")

    # Check the output dimensions
    assert output.shape[0] == 1 and output.shape[1] == 2, "The model output does not have two values"

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middlewares
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
