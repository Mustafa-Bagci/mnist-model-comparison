import idx2numpy
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn import tree, ensemble, neighbors, naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay
import os
import pandas as pd

# MNIST IDX dataset paths
mnist_folder = './MNIST Datasets'

train_images_path = os.path.join(mnist_folder, 'train-images.idx3-ubyte')
train_labels_path = os.path.join(mnist_folder, 'train-labels.idx1-ubyte')
test_images_path = os.path.join(mnist_folder, 't10k-images.idx3-ubyte')
test_labels_path = os.path.join(mnist_folder, 't10k-labels.idx1-ubyte')

# Check if files exist
for path in [train_images_path, train_labels_path, test_images_path, test_labels_path]:
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        exit()

# Load IDX files into numpy arrays
X_train = idx2numpy.convert_from_file(train_images_path).astype('float32') / 255.0
y_train = idx2numpy.convert_from_file(train_labels_path)
X_test = idx2numpy.convert_from_file(test_images_path).astype('float32') / 255.0
y_test = idx2numpy.convert_from_file(test_labels_path)

# Reshape images for CNN model
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Flatten images for traditional ML models
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# CNN model
cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the CNN model
cnn_history = cnn_model.fit(X_train, y_train, epochs=10, batch_size=64,
                            validation_data=(X_test, y_test),
                            callbacks=[early_stopping], verbose=1)

# Machine learning models
models_dict = {
    "J48": tree.DecisionTreeClassifier(max_depth=10),
    "KNN": neighbors.KNeighborsClassifier(n_neighbors=5),
    "Random Forest": ensemble.RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Naive Bayes": naive_bayes.GaussianNB(),
    "Decision Tree": tree.DecisionTreeClassifier(max_depth=10)
}

model_names = []
accuracies = []
auc_scores = []
precisions = []
recalls = []
f1_scores = []
mcc_scores = []

output_folder = './Output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Train and evaluate traditional ML models
for name, model in models_dict.items():
    print(f"Training {name} model...")
    model.fit(X_train_flat, y_train)
    y_pred = model.predict(X_test_flat)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')

    cm_path = os.path.join(output_folder, f"{name}_confusion_matrix.png")
    plt.title(f'{name} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"{name} Confusion Matrix saved as '{cm_path}'.")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_test, y_pred)

    try:
        if hasattr(model, "predict_proba"):
            y_pred_prob = model.predict_proba(X_test_flat)
            auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
        else:
            auc = np.nan
    except ValueError as e:
        print(f"AUC calculation error for {name} model: {e}")
        auc = np.nan

    model_names.append(name)
    accuracies.append(accuracy)
    auc_scores.append(auc)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    mcc_scores.append(mcc)

# CNN confusion matrix
y_pred_cnn = cnn_model.predict(X_test)
y_pred_cnn = np.argmax(y_pred_cnn, axis=1)

cm_cnn = confusion_matrix(y_test, y_pred_cnn)
disp_cnn = ConfusionMatrixDisplay(confusion_matrix=cm_cnn, display_labels=range(10))
disp_cnn.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')

cm_cnn_path = os.path.join(output_folder, "CNN_confusion_matrix.png")
plt.title('CNN Confusion Matrix')
plt.tight_layout()
plt.savefig(cm_cnn_path)
plt.close()
print(f"CNN Confusion Matrix saved as '{cm_cnn_path}'.")

cnn_accuracy = accuracy_score(y_test, y_pred_cnn)
cnn_precision = precision_score(y_test, y_pred_cnn, average='weighted', zero_division=1)
cnn_recall = recall_score(y_test, y_pred_cnn, average='weighted')
cnn_f1 = f1_score(y_test, y_pred_cnn, average='weighted')
cnn_mcc = matthews_corrcoef(y_test, y_pred_cnn)
cnn_auc = np.nan

model_names.append("CNN")
accuracies.append(cnn_accuracy)
auc_scores.append(cnn_auc)
precisions.append(cnn_precision)
recalls.append(cnn_recall)
f1_scores.append(cnn_f1)
mcc_scores.append(cnn_mcc)

# Save results to an Excel file
results_df = pd.DataFrame({
    "Model": model_names,
    "Accuracy": accuracies,
    "AUC": auc_scores,
    "Precision": precisions,
    "Recall": recalls,
    "F1 Score": f1_scores,
    "MCC": mcc_scores
})

excel_path = os.path.join(output_folder, "model_comparison.xlsx")
with pd.ExcelWriter(excel_path) as writer:
    results_df.to_excel(writer, sheet_name='Model Metrics', index=False)

print(f"Results saved to '{excel_path}'.")
