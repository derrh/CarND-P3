import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# Load the data samples
samples = []
with open('driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    samples.append(line)

def preprocess_image(path):
  # Loads the image at the given path and normalizes/mean-centers the data
  file_name = path.split('/')[-1]
  current_path = 'IMG/' + file_name
  image = cv2.imread(current_path) 
  # I had trouble with the lambda layer on my mac after training on an aws
  # instance, so I preprocessed out-of-network
  return image.astype(np.float) / 255.0 - 0.5

def reverse_image(image):
  # Mirrors the input image around the y axis
  r = image.copy()
  r = cv2.flip(image, 1)
  return r

# generator to load images and yield them in batches
STEERING_CORRECTION = 0.2
def generator(samples, batch_size=32):
  # yields batches of feature data from the given samples data set
  num_samples = len(samples)
  try:
    while 1: # Loop forever so the generator never terminates
      shuffle(samples)
      for offset in range(0, num_samples, batch_size):
        batch_samples = samples[offset:offset+batch_size]
  
        images = []
        angles = []
        for batch_sample in batch_samples:
          center = preprocess_image(batch_sample[0])
          images.append(center)
          steering = float(batch_sample[3])
          angles.append(steering)
  
          images.append(reverse_image(center))
          angles.append(-steering)
        
          left = preprocess_image(batch_sample[1])
          images.append(left)
          angles.append(steering + STEERING_CORRECTION)
        
          images.append(reverse_image(left))
          angles.append(-(steering + STEERING_CORRECTION))
        
          right = preprocess_image(batch_sample[2])
          images.append(right)
          angles.append(steering - STEERING_CORRECTION)
          
          images.append(reverse_image(right))
          angles.append(-(steering - STEERING_CORRECTION))
        
        X_train = np.array(images)
        y_train = np.array(angles)
        yield shuffle(X_train, y_train)
  except GeneratorExit:
    print('Finished generating samples')    
  finally:
    print('Finally... sigh')

# Split the dataset and instantiate the generators
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# NVIDIA's self-driving car reference model
# https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

model.compile(
  loss='mse',
  optimizer='adam'
)
model.fit_generator(
  train_generator,
  samples_per_epoch=len(train_samples),
  validation_data=validation_generator,
  nb_val_samples=len(validation_samples),
  nb_epoch=8
)

print("saving model!")
model.save('model.h5')
exit()

