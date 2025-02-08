# MNIST Model Comparison

This project compares different machine learning and deep learning models for handwritten digit classification using the MNIST dataset. It evaluates models such as CNN, Decision Trees, KNN, Random Forest, and Naive Bayes.

## 📌 Features
- Loads and preprocesses the MNIST dataset.
- Implements a Convolutional Neural Network (CNN) for classification.
- Trains and evaluates multiple machine learning models:
  - Decision Tree (J48 & Standard)
  - K-Nearest Neighbors (KNN)
  - Random Forest
  - Naive Bayes
- Calculates key performance metrics:
  - Accuracy
  - AUC (Area Under Curve)
  - Precision
  - Recall
  - F1 Score
  - Matthews Correlation Coefficient (MCC)
- Generates and saves confusion matrices for each model.
- Saves results in an Excel file for easy analysis.

## 📥 Download Dataset
Before running the model, download the MNIST dataset using the following command:

```sh
python download_data.py
```

This script will automatically download and place the dataset in the correct folder.

## 🛠 Setup and Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/Mustafa-Bagci/mnist-model-comparison.git
   cd mnist-model-comparison
   ```
2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Download the dataset using:
   ```sh
   python download_data.py
   ```

## 🚀 Usage
Run the script to train models and save results:
```sh
python main.py
```
After execution, results will be saved in the `Output` folder, including:
- `model_comparison.xlsx` – Performance metrics for all models
- `cnn_accuracy_plot.png` – CNN training accuracy plot
- Confusion matrices for each model

## 📊 Results
The models are evaluated based on accuracy, precision, recall, F1-score, and AUC. Results are stored in an Excel file for easy comparison.

## 📜 License
This project is licensed under the MIT License.

## 👨‍💻 Author
Developed by [Mustafa Enes Bagci](https://github.com/Mustafa-Bagci). Feel free to contribute or report issues!