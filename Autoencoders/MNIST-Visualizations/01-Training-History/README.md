# MNIST Autoencoder Visualizations 01

This is a project I did in July 2022. Some of the results can be seen below.

The first image here is a gif showing the training history of a single digit. In this case you can see that as training progressed, the digit became more clear. In this case it was a seven.

<img src="https://github.com/Kinvert/Machine-Learning/blob/master/Autoencoders/MNIST-Visualizations/01-Training-History/single_digit.gif" width="280" height="280"/>

The second image shows basically the same animation as above, except rather than animating it, the frames are arranged in a tiled image. This was a different digit, in this case a 6.

<img src="https://github.com/Kinvert/Machine-Learning/blob/master/Autoencoders/MNIST-Visualizations/01-Training-History/train_history.png" width="280" height="280"/>

## YouTube Video

I briefly talk about this project here:

[![Discussion Video](https://img.youtube.com/vi/L52H2fggL5U/0.jpg)](https://www.youtube.com/watch?v=L52H2fggL5U&t=16s "Discussion Video")

## MP4s

The single_digit.mp4 in this folder is similar to the first image, which is a gif. In this case, I created an HTML5 video. It doesn't show up in the ipynb in GitHub so I saved it as an MP4 and included it in the folder.

The tiled_video.mp4 shows a specific digit in each row, and how it progressed as there were more training epochs. Each digit starts out very messy but over time the autoencoder learns to more accurately recreate each digit.
