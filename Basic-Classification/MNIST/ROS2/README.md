# ROS2 MNIST

Basically the plan is to train it on MNIST and then use ROS2 to communicate with a webcam, I can draw numbers and it should run inference.

![1](<Screenshot from 2025-08-26 01-32-38.png>)

![2](<Screenshot from 2025-08-26 01-35-57.png>)

TODO:
- double check image output matches model input
- better image processing
- improve frame rate
- add escape to keys for closing windows
- train on augmentation
- some sort of cool visualizations
- somehow try to get this in the browser
- clean up code
- improve model output location, avoid steps like copying to install etc
- try something cool like sort of a GAN taking the input and generating a "good" number from it
- find a way to incorporate PufferLib
- auto detect numbers rather than the ROI
