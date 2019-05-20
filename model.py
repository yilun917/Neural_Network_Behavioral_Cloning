import cv2
import csv
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Input
from keras.layers import Convolution2D, Cropping2D, GlobalAveragePooling2D, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.applications.inception_v3 import InceptionV3
import matplotlib.pyplot as plt

#data processing
data_path = "/opt/carnd_p3/data/"
lines = []
with open(data_path+"driving_log.csv") as csvfile:   # read in csv file
    reader = csv.reader(csvfile)
    header = next(reader)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []
correction = 0.2
# read in the data and store in variable
for line in lines:
    source_path = line[0]
    filename= source_path.split("/")[-1]
    current_path = data_path + source_path
    #current_path = data_path +"IMG/"+ filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    #left camera image
    source_path1 = line[1]
    filename1 = source_path1.split("/")[-1]
    current_path1 = data_path + "IMG/" + filename1
    image1 = cv2.imread(current_path1)
    images.append(image1)
    measurement1 = float(line[3]) + correction
    measurements.append(measurement1)
    #right camera image
    source_path2 = line[2]
    filename2 = source_path2.split("/")[-1]
    current_path2 = data_path + "IMG/" + filename2
    image2 = cv2.imread(current_path2)
    images.append(image2)
    measurement2 = float(line[3]) - correction
    measurements.append(measurement2)
    
    
#augment the image and measurement
#add the original image and the flipped image as well as the flipped steering data together
augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*(-1))

#convert the list to numpy array and checking the size    
X_train = np.array(augmented_images)
print(X_train.shape)
y_train = np.array(augmented_measurements)
print(y_train.shape)



#build basic NN, structure of Nvidea
model = Sequential()
model.add(Lambda(lambda x: x/255-0.5, input_shape=(160,320,3)))  # data noamalization
#convert image to gray
#Source: https://stackoverflow.com/questions/46836358/keras-rgb-to-grayscale
model.add(Lambda(lambda x: (0.33 * x[:,:,:,:1]) + (0.33 * x[:,:,:,1:2]) + (0.33 * x[:,:,:,-1:])))
model.add(Cropping2D(cropping=((50,20),(0,0))))  # cropping the top and bottom
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))   #the published NVIDIA model
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss="mse", optimizer='adam')
history_object  = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5, verbose = 2)  #create history object to do model visualization later
#model.save('model.h5')

#output drain history
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig("Training_History.jpg")
plt.show()

