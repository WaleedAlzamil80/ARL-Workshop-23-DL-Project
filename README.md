# Project Description

In this project, we use a neural network to clone car driving behaviour.  It is a supervised regression problem between the car steering angle, throttle, and the road images in front of the car.

Those images were taken from three different camera angles (from the centre, the left, and the right of the car).

The goal is to build a Neural Network Model using [Pytorch](https://pytorch.org/) Fram Work that takes the three Images and outputs the steering angle and the throttle the car must take

## What should you do??

- Clone this repo.
- Download the Simulator from here [windows-64](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f3a4_simulator-windows-64/simulator-windows-64.zip) [windows-32](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f4b6_simulator-windows-32/simulator-windows-32.zip) [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip) [Mac](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip).
- You will need to collect your own data from the simulator as illustrated [Here](https://drive.google.com/file/d/12H6iWTMtMLTnDe89tX1HcXHGgKKbQZbg/view?usp=drive_link) and use it to train your own model.
- You will need to modify the `model.py` file only
- In the `model.py` file You will need to add the architecture of your model.
- Your Model must be saved using `torch.save(model, "model.pth")` and must be in the same directory.

# How to run
After Cloning this repo and you have done all the previous tasks Open the terminal and run 

`python drive.py model.pth images`
 
Then open the simulator in Autonomous mode. Now you can see the performance of your trained model :)

# Some Issues
maybe you will face some problems with the dependence on the `socketio` you can try the following
```
pip install --upgrade python-socketio==4.6.0

pip install --upgrade python-engineio==3.13.2

pip install --upgrade Flask-SocketIO==4.3.1
```

# References
- NVIDIA model: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
- Udacity Self-Driving Car Simulator: https://github.com/udacity/self-driving-car-sim
