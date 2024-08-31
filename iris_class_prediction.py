# Implementation of the Model in Python
# Stage 1 - Building the CNN
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Stage 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255, width_shift_range = 0.2,    
                                   height_shift_range = 0.2, shear_range = 0.2,
                                   zoom_range = 0.2)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'binary')
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

history = classifier.fit(training_set, steps_per_epoch = 404, epochs = 20,
                        validation_data = test_set, validation_steps = 101)

# Stage 3 - Making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/left/left or right4.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'right iris'
else:
    prediction = 'left iris'

print(prediction)

# Stage 4 - Evaluating the model and plotting graphs using matplotlib
_, acc = classifier.evaluate(test_set, steps=len(test_set), verbose=0)
print('> %.3f' % (acc * 100.0))

# Confusion Matrix and Classification Report
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

test_set = test_datagen.flow_from_directory('dataset/test_set’,
                                            target_size = (64, 64),
						                    shuffle = False,
                                            batch_size = 32,
                                            class_mode = 'binary')

Y_pred = classifier.predict_classes(test_set, batch_size=None, verbose=0)
Y_pred = (Y_pred > 0.5).astype(int)
print('Confusion Matrix')
print(confusion_matrix(test_set.classes, Y_pred))
print('Accuracy Score: ',accuracy_score(test_set.classes, Y_pred) * 100,'%')
print('Classification Report')
target_names = [‘left’, ‘right’]
print(classification_report(test_set.classes, Y_pred, target_names=target_names))

#Evaluating Accuracy and Loss for the Model
# Retrieve a list of accuracy results on training and validation data sets for each training epoch
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Retrieve a list of list results on training and validation data sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')
plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')
