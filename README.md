Behavioral Cloning
====

**By Derrick Hathaway<br />
December 12, 2017**

<br />

## Files

- **README.md** This writeup
- **model.py** The code to train the network
- **drive.py** The code to drive the car in the simulator 
- **model.h5** The trained model 
- **run1.mp4** A video of the simulator being driven by the trained model

<br />

## Model Architecture and Training Strategy

I chose to use the model architecture NVIDIA described in [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). This model is sequential and has the following layers:

- Input: 160x320x3
- Croping2D: 65x320x3
- 5x5 2D Convolution: 31x158x24
- 5x5 2D Convolution: 14x77x36
- 5x5 2D Convolution: 5x37x48
- 3x3 2D Convolution: 3x35x64
- 3x3 2D Convolution: 1x33x64
- Flatten: 2112
- Fully Connected: 100
- Fully Connected: 50
- Fully Connected: 1

You will notice that the normalization and mean-centering of the data is not done in the model. This is because I was having issue with the Lamda layer on my mac where I was running the driving simulator. The `model.py` training program normalizes and mean-centers the data as the images are loaded. The `drive.py` has also been updated to normalize and mean-center the data.

In order to reduce overfitting I used separate training, validation and test data. I separated the gathered data into two sets for training and validation and then tested the model in the simulator. The training data was also shuffled before training.

The training took place over 8 epochs, and I used an Adam optimizer which internally chooses and adjusts the learning rate.

The training data was gathered via the driving simulator. I drove the car in the simulator on the first test track using my mouse. I used about 2 full laps worth of data, plus several sets of data collected while driving around difficult corners and areas where the dirt track branches from the paved track. I augmented this data set by using both the center image and the left and right camera images with slight steering corrections. In addition, I mirrored each image and reversed it's steering angle. Using these two strategies I was able to turn one steering measurement with three camera images into 6 data points for training.

<br />

## Training

I found it very useful to iterate on the model. This allowed me to very quickly verify that I could collect data, train the model and use that model to drive the car in the simulator. For instance I started out with a very simple model with just two fully connected layers. I trained these earlier models on a couple of laps worth of data. These models were not capable of extracting the features necessary to achieve a high degree of accuracy. The mean squared error for the test and validation data sets ranged from, on the simple model with two fully connected layers, being on the order of thousands, to an attempt to use the LeNet architecture, being on the order of tenths to hundreths of a degree. Even the accurancy of LeNet was not sufficient to keep the car on the road around tight corners.

After trying these simpler architectures I decided to try the NVIDIA reference architecture cited earlier. This improved the accuracy dramatically and both the training and validation loss was on the order of thousanths of a degree.

One key step I took to improve performance led to a major bug that took several hours to discover and moments to fix. I decided to normalize and mean-center the input data. This made a big difference in the training and validation loss, but the strategy I tried initially, a `Lambda` layer with the simple equation `x / 255 - 0.5`, worked perfectly on the AWS instance I was using to train, but would crash when I loaded the model locally on my Mac. Not being a Python guru, I decided the simplest solution would be to adjust the data as the image was loaded rather than doing it in the model. I did not consider at the time that the `drive.py` would also have to perform the adjustment before feeding the data from the simulator to the model during automated driving. After several hours of training and retraining and after trying dropout, adding more data, and other strategies, I finally noticed that the steering in the simulator that my model was predicting always went to the extreme values: from `-25` to `25` and quickly back to `-25`. It rarely steered at a small angle near zero. Finally the moment of realization dawned on me. The input values provided by the simulator during automated driving were in the range `[0, 255]` but the model was expecting values in the range `[-0.5, 0.5]`. After correcting this issue, I was able to make progress and eventually train a model that could complete laps successfully without leaving the road.

During training I also found that I needed more data. I was training on about 2 full laps worth of data. The first strategy I used was to flip the center image horizontally (around the y axis) in order to double my data set. Later I eventually added the left and right images with a slight steering correction toward the center.

<br />

## Simulation

In the included [video file](run1.mp4) you can see that the model is capable of driving the car around the entire track without the vehicle leaving the safe driving area.
