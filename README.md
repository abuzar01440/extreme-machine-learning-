# 🚀 Extreme Learning Machine (ELM) Implementation

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> A comprehensive implementation of Extreme Learning Machines using **TensorFlow** for ultra-fast neural network training

## 🧠 What is Extreme Learning Machine?

Extreme Learning Machine (ELM) is a novel learning algorithm for single-hidden layer feedforward neural networks (SLFNs) that:

- ⚡ **Ultra-fast training** - thousands of times faster than traditional neural networks
- 🎯 **Good generalization performance**
- 🔄 **No iterative tuning** - input weights are randomly assigned and never updated
- 📊 **Analytical solution** - output weights are calculated in a single step using pseudo-inverse

### Key Characteristics:

- **Random Initialization**: The input weights and biases are randomly initialized and never updated during training
- **Non-iterative Training**: No backpropagation needed, eliminating the slow gradient-based learning
- **Analytical Learning**: Output weights calculated directly using matrix operations
- **Versatile Applications**: Works for both regression and classification tasks

## 🛠️ Built With

- **Python 3.11** - Core programming language
- **TensorFlow** - Primary framework for all algorithm implementations
- **NumPy** - Supporting numerical operations

## 📋 Repository Contents

This repository contains multiple implementations of ELM, all built with **TensorFlow**:

1. **Basic ELM**: Foundation implementation of the Extreme Learning Machine algorithm
2. **ELM Regressor**: Specialized implementation for regression problems
3. **ELM Classifier**: Specialized implementation for classification problems

## 💻 TensorFlow Implementations

### Basic ELM Implementation in TensorFlow

```python
class ELM:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Randomly initialize input weights and biases (key ELM aspect)
        self.input_weights = tf.Variable(tf.random.normal([input_size, hidden_size]))
        self.biases = tf.Variable(tf.random.normal([hidden_size]))

        # Output weights will be learned
        self.output_weights = tf.Variable(tf.zeros([hidden_size, output_size]))

    def hidden_layer(self, X):
        X = tf.cast(X, tf.float32)  # Cast X to float32
        # Activation function
        H = tf.nn.sigmoid(tf.matmul(X, self.input_weights) + self.biases)
        return H

    def train(self, X, y):
        H = self.hidden_layer(X)
        # Solve for output weights using pseudo-inverse
        y = tf.cast(y, tf.float32)
        self.output_weights.assign(tf.linalg.pinv(H) @ tf.reshape(y, [-1, 1]))

    def predict(self, X):
        H = self.hidden_layer(X)
        y_pred = tf.matmul(H, self.output_weights)
        return y_pred
```

### ELM Regressor with TensorFlow

```python
class ELMRegressor_TF:
    def __init__(self, n_hidden_units):
        self.n_hidden_units = n_hidden_units

    def fit(self, X, labels):
        # Convert inputs to tensors
        X = tf.constant(X, dtype=tf.float32)
        labels = tf.constant(labels, dtype=tf.float32)

        # Add bias column (ones)
        X = tf.concat([X, tf.ones([tf.shape(X)[0], 1], dtype=tf.float32)], axis=1)

        # Initialize random weights and calculate G (hidden layer output)
        self.random_weights = tf.Variable(
            tf.random.normal([tf.shape(X)[1], self.n_hidden_units], dtype=tf.float32)
        )
        G = tf.nn.tanh(tf.matmul(X, self.random_weights))

        # Compute output weights (w_elm) using pseudo-inverse
        self.w_elm = tf.matmul(tf.linalg.pinv(G), labels)

    def predict(self, X):
        # Convert input and add bias
        X = tf.constant(X, dtype=tf.float32)
        X = tf.concat([X, tf.ones([tf.shape(X)[0], 1], dtype=tf.float32)], axis=1)

        # Calculate hidden layer output and predictions
        G = tf.nn.tanh(tf.matmul(X, self.random_weights))
        return tf.matmul(G, self.w_elm)
```

### ELM Classifier with TensorFlow

```python
class ELM_classifier:
    def __init__(self, input_size, output_size, hidden_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # TensorFlow Variable initialization for weights and bias
        self.weight = tf.Variable(
            tf.random.uniform([self.hidden_size, self.input_size], -1, 1, dtype=tf.float32)
        )
        self.bias = tf.Variable(
            tf.random.uniform([self.hidden_size], -1, 1, dtype=tf.float32)
        )
        self.beta = None  # Will be initialized during training

    def sigmoid(self, x):
        return tf.math.sigmoid(x)  # TensorFlow's sigmoid activation

    def predict(self, X):
        X = tf.cast(X, tf.float32)
        H = self.sigmoid(tf.matmul(X, tf.transpose(self.weight)) + self.bias)
        return tf.matmul(H, self.beta)

    def train(self, X, y):
        X = tf.cast(X, tf.float32)
        y = tf.cast(y, tf.float32)
        y = tf.reshape(y, [-1, 1])  # Ensure y is a column vector

        # Calculate hidden layer output
        H = self.sigmoid(tf.matmul(X, tf.transpose(self.weight)) + self.bias)

        # Compute beta (output weights) using TensorFlow's pseudo-inverse
        H_pinv = tf.linalg.pinv(H)
        self.beta = tf.matmul(H_pinv, y)

        # Return training predictions (optional, for evaluation)
        return tf.matmul(H, self.beta)
```

## 🔬 Example Use Cases

### Regression Example with TensorFlow

```python
# Sample data
X = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)
labels = tf.constant([[10], [20], [30]], dtype=tf.float32)

# Create and train the model
elm_regressor = ELMRegressor_TF(n_hidden_units=10)
elm_regressor.fit(X, labels)

# Make predictions
new_X = tf.constant([[7, 8], [9, 10]], dtype=tf.float32)
predictions = elm_regressor.predict(new_X)
print(predictions)
```

### Classification Example with TensorFlow

```python
X_train = np.array([
    [0.85, 0.32, 0.68],  # Features for class 1
    [0.12, 0.91, 0.25],  # Features for class 0
    [0.79, 0.45, 0.55],  # Features for class 1
    [0.21, 0.88, 0.33],  # Features for class 0
    [0.92, 0.28, 0.71]   # Features for class 1
])

y_train = np.array([1, 0, 1, 0, 1])  # Binary labels

# Create and train the classifier
elm_classifier = ELM_classifier(input_size=3, output_size=1, hidden_size=10)
elm_classifier.train(X_train, y_train)

# Make predictions
predictions = elm_classifier.predict(X_test).numpy()
for i, pred in enumerate(predictions):
    if pred > 0.5:
        print(f"Sample {i+1}: Class 1 (Probability: {pred[0]:.2f})")
    else:
        print(f"Sample {i+1}: Class 0 (Probability: {1-pred[0]:.2f})")
```

## 📈 Advantages of ELM

- 🏎️ **Speed**: Training is extremely fast compared to traditional neural networks
- 🧩 **Simplicity**: No hyperparameters to tune for the training algorithm itself
- 🔀 **Deterministic Results**: For a given random initialization, results are deterministic
- 🛠️ **Universal Approximation**: Theoretically proven to have universal approximation capability
- 💪 **Robust Performance**: Often competitive with more complex models

## 👨‍💻 About the Author

**Abuzar Shahid**  
Machine Learning Researcher focused on efficient neural network architectures.

📧 Email: abuzarbhutta@gmail.com  
🔗 GitHub: [abuzar01440](https://github.com/abuzar01440)

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

⭐ If you find this **TensorFlow-based** implementation useful, please consider giving it a star! ⭐
