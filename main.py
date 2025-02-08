import idx2numpy
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

# Define the correct MNIST file paths
mnist_folder = "./MNIST Datasets"
train_images_path = os.path.join(mnist_folder, "train-images.idx3-ubyte")
train_labels_path = os.path.join(mnist_folder, "train-labels.idx1-ubyte")
test_images_path = os.path.join(mnist_folder, "t10k-images.idx3-ubyte")
test_labels_path = os.path.join(mnist_folder, "t10k-labels.idx1-ubyte")

# Function to load IDX files
def load_idx_data(image_path, label_path):
    images = idx2numpy.convert_from_file(image_path).astype("float32") / 255.0  # Normalize
    labels = idx2numpy.convert_from_file(label_path)
    return images, labels

# Load train and test data
X_train, y_train = load_idx_data(train_images_path, train_labels_path)
X_test, y_test = load_idx_data(test_images_path, test_labels_path)

# Reshape images to (28,28,1) for CNN
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print(f"Training data shape: {X_train.shape}, Labels shape: {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, Labels shape: {y_test.shape}")

# Create CNN model
cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # Output layer with softmax for 10 classes
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
cnn_model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Early stopping callback function
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
cnn_history = cnn_model.fit(X_train, y_train, epochs=10, batch_size=64,
                            validation_data=(X_test, y_test),
                            callbacks=[early_stopping], verbose=1)

# Evaluate the model on test data
test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test, verbose=0)

# Save results to a DataFrame
epochs = range(1, len(cnn_history.history['loss']) + 1)
cnn_results = pd.DataFrame({
    'Epoch': epochs,
    'Loss': cnn_history.history['loss'],
    'Accuracy': cnn_history.history['accuracy'],
    'Validation Loss': cnn_history.history['val_loss'],
    'Validation Accuracy': cnn_history.history['val_accuracy'],
})

# Add test results to the DataFrame
test_results = pd.DataFrame({
    'Epoch': ['Test Results'],
    'Loss': [f"{test_loss:.5f}"],
    'Accuracy': [f"{test_accuracy:.4f}"],
    'Validation Loss': [None],
    'Validation Accuracy': [None],
})
cnn_results = pd.concat([cnn_results, test_results], ignore_index=True)

# Check and create the output folder
output_folder = './Output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Save results to an Excel file (in the Output folder)
cnn_results.to_excel(os.path.join(output_folder, "cnn_results.xlsx"), index=False)

# Plot training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(cnn_results['Epoch'][:-1], cnn_results['Accuracy'][:-1], label='Training Accuracy', marker='o')
plt.plot(cnn_results['Epoch'][:-1], cnn_results['Validation Accuracy'][:-1], label='Validation Accuracy', marker='o')
plt.title('CNN Model Accuracy (Across Epochs)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()

# Save the plot as a PNG file in the Output folder
plt.savefig(os.path.join(output_folder, 'cnn_accuracy_plot.png'))
plt.show()

# Compute and plot the Confusion Matrix
y_pred = cnn_model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)  # Predicted classes

cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')

# Save the confusion matrix plot
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'cnn_confusion_matrix.png'))
plt.show()

print("CNN results saved to 'Output/cnn_results.xlsx'.")
print("Accuracy plot saved as 'Output/cnn_accuracy_plot.png'.")
print("Confusion matrix visualization saved as 'Output/cnn_confusion_matrix.png'.")
