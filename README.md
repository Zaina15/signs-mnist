# 🧠 Sign Language Recognition with Neural Networks

This project uses a **Multi-layer Perceptron (MLP) Classifier** from `scikit-learn` to recognize American Sign Language (ASL) hand signs based on image pixel data. It is trained and tested on a subset of the **SignMNIST** dataset.

---

## 📁 Project Structure
sign-language-mlp/

├── main.py # Main training, testing, and evaluation script

├── sign_mnist_13bal_train.csv # Training dataset (balanced)

├── sign_mnist_13bal_test.csv # Test dataset (balanced)

└── README.md


---

## 🚀 How It Works

- **Data Loading**: Reads CSV files containing labeled image data.
- **Preprocessing**: Normalizes pixel values to [0, 1].
- **Model**: A simple neural network with one hidden layer (`hidden_layer_sizes=(40)`).
- **Training**: The model is trained on the training set using the `MLPClassifier`.
- **Evaluation**:
  - Prints class-wise and overall accuracy on the **test** dataset.
  - Displays the **confusion matrix**.
  - Identifies the **top 3 most commonly misclassified letter pairs**.

---

## 🛠️ Requirements

- Python 3.8+
- `pandas`
- `scikit-learn`

Install dependencies:

```bash
pip install pandas scikit-learn
```

📚 Dataset Source: https://www.kaggle.com/datasets/datamunge/sign-language-mnist
