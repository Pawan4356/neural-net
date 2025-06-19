#  🚀 Neural Network from Scratch

This repository implements a simple Neural Network (NN) from scratch using Python and NumPy. It demonstrates core concepts like forward propagation, backpropagation, and training with gradient descent.

<div align="center">
  <img src="NeuralNetwork.png" alt="NN" width="500">
</div>

## 🧠 What is a Neural Network?

A **Neural Network** is a computational model inspired by the human brain. It consists of interconnected layers of "neurons" where:

- **Input layer** receives the raw data.
- **Hidden layers** process and learn patterns.
- **Output layer** makes predictions.

Each neuron applies a weighted sum and passes it through an **activation function** to introduce non-linearity.


## 📚 How Learning Works

1. **Forward Propagation**  
   Input → Weighted Sum → Activation → Output

2. **Loss Calculation**  
   Compare predicted output with true labels using a loss function.

3. **Backpropagation**  
   Compute gradients of the loss w.r.t. weights using the chain rule.

4. **Weight Update**  
   Update weights using **Gradient Descent** to minimize the loss.


## 📊 Example Use Cases

- Predicting handwritten digits (MNIST)
- Binary classification problems (e.g., spam vs not spam)
- Simple regression tasks

## 🛠️ Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/your-username/neural-network.git
cd neural-network
```

2. **Create a Virtual Environment (Optional)**

```bash
python -m venv NN
# Activate the environment
# Windows:
NN\Scripts\activate
# Linux/macOS:
source NN/bin/activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```
