import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data_dir = 'data'
classes = ['0', 'A', 'B', 'C']
img_size = (32, 32)
data = []
label = []

# Load data and prepare images
for index, name_class in enumerate(classes):
    path = os.path.join(data_dir, name_class)
    for image_name in os.listdir(path):
        image_path = os.path.join(path, image_name)
        # Load
        image = cv2.imread(image_path)
        image = image[...,::-1]
        # Resize
        image = cv2.resize(image, img_size)
        data.append(image)
        label.append(index)
        # Augmentacja
        for x in [-1, 0, 1]:
            image = cv2.flip(image,x)
            data.append(image)
            label.append(index)         

# Normalize
data = np.array(data) /255
label = np.array(label)

cv2.destroyAllWindows()  # Close all windows after displaying all images
# splitting data 80% - training, 10% - validate, 10% - test
training_data, res_data, training_label, res_label = train_test_split(data, label, test_size=0.2, random_state=42)
validate_data, test_data, validate_label, test_label = train_test_split(res_data, res_label, test_size=0.5, random_state=42)

# Create model
model = Sequential()

# Add first convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)))
model.add(MaxPooling2D((2, 2)))

# Add second convolutional layers
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Add layer Dropout
model.add(Dropout(0.5))

# Add layer Flatten
model.add(Flatten())

# Add two layers Dense
model.add(Dense(128, activation='relu'))
model.add(Dense(4, activation='softmax'))

# Show model
model.summary()

# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer=RMSprop(lr=0.0001), metrics=['accuracy'])

# Fit model
history = model.fit(training_data, training_label , epochs=20, validation_data=[validate_data, validate_label], batch_size = 128)

# Saving the model
model.save('TOMASZ_ZIETKOWSKI_model.h5')

# Plotting the learning curves
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

# Testing in test_data
test_result = model.predict(test_data)
test_pred = np.argmax(test_result, axis=1)

print(classification_report(test_label, test_pred))

# Show - 10 random images
random_indexes = np.random.choice(len(test_data), size=10, replace=False)
predictions = model.predict(test_data[random_indexes])

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(test_data[random_indexes[i]])
    true_label = test_label[random_indexes[i]]
    pred_label = np.argmax(predictions[i])
    ax.set_title(f"True label: {true_label}, Pred label: {pred_label}")
    ax.axis('off')

plt.show()
print()
