# 🧠 Deep Learning Practice Repository

A hands-on deep learning practice collection using TensorFlow and Keras with real-world datasets for classification, regression, clustering, and computer vision tasks.

## ✨ Features

- **Interactive Jupyter Notebooks**: Ready-to-run notebooks covering core deep learning concepts
- **Classification & Regression**: Practical implementations with Fashion MNIST and other datasets
- **Clustering Algorithms**: Weather data clustering and unsupervised learning examples
- **Convolutional Neural Networks (CNNs)**: Image recognition and digit classification models
- **Example Datasets**: Pre-configured notebooks with MNIST, Fashion MNIST, and custom datasets
- **TensorFlow Basics**: Foundational tutorials on tensors, operations, and model building

## 🚀 Quick Start/Installation

### Clone the repository
```bash
git clone https://github.com/Nikshay-Jain/Deep-Learning.git
cd Deep-Learning
```

### Install dependencies
```bash
pip install -r requirements.txt
```
*If `requirements.txt` is not available, install core dependencies:*
```bash
pip install tensorflow jupyter numpy pandas matplotlib scikit-learn
```

### Launch Jupyter Notebook
```bash
jupyter notebook
```

## 🛠️ Usage

1. **Open Jupyter Notebook** in your browser (automatically launches at `http://localhost:8888`)
2. **Navigate to any notebook** from the file browser:
   - `Tensorflow Basics.ipynb` - Start here for TensorFlow fundamentals
   - `Deep Neural Networks - Fashion MNIST.ipynb` - Classification example
   - `Convolutional Neural Networks.ipynb` - CNN implementation
   - `Clustering - Weather.ipynb` - Unsupervised learning example
3. **Run cells sequentially** using `Shift + Enter`

### Example: Running a Classification Model
```python
# Inside any notebook
import tensorflow as tf
from tensorflow import keras

# Load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build and train model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
```

---

**💡 Tip**: Explore the `Classification` and `Regression` directories for organized topic-specific notebooks.
