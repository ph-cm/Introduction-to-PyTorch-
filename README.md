# Neural Frameworks

## 📚 Overview

We have learned that to train neural networks, you need:

- Quickly multiply matrices (tensors)
- Compute gradients to perform gradient descent optimization

### What Neural Network Frameworks Allow You to Do:
- Operate with tensors on whatever compute is available: CPU, GPU, or even TPU.
- Automatically compute gradients (they are explicitly programmed for all built-in tensor functions).

### Additional Features (Optional):
- Neural Network constructor / higher-level API (describe network as a sequence of layers).
- Simple training functions (like `fit` in Scikit-Learn).
- A number of optimization algorithms in addition to gradient descent.
- Data handling abstractions (that will ideally work on GPU, too).

---

## 🚀 Most Popular Frameworks

1. **TensorFlow 1.x** - The first widely available framework (Google). Allowed defining static computation graphs, pushing them to GPUs, and explicitly evaluating them.

2. **PyTorch** - A framework from Facebook that is growing in popularity.

3. **Keras** - A higher-level API on top of TensorFlow/PyTorch to unify and simplify using neural networks (created by François Chollet).

4. **TensorFlow 2.x + Keras** - New version of TensorFlow with integrated Keras functionality, supporting dynamic computation graphs, allowing tensor operations very similar to NumPy (and PyTorch).

---

## 🧠 Basic Concepts: Tensor

A **tensor** is a multi-dimensional array. It is very convenient to use tensors to represent different types of data:

- `400x400` - black-and-white picture
- `400x400x3` - color picture
- `16x400x400x3` - minibatch of 16 color pictures
- `25x400x400x3` - one second of 25-fps video
- `8x25x400x400x3` - minibatch of 8 1-second videos

### Simple Tensors
You can easily create simple tensors from lists of NumPy arrays, or generate random ones.

---

## 🔧 In-place and Out-of-place Operations

Tensor operations such as `+`/`add` return new tensors. However, sometimes you need to modify the existing tensor in-place. Most of the operations have their in-place counterparts, which end with `_`.

---

## 🔄 Computing Gradients

For backpropagation, you need to compute gradients. By setting any PyTorch Tensor's attribute `requires_grad` to `True`, all operations with this tensor will be tracked for gradient calculations. To compute the gradients, you need to call the `backward()` method, after which the gradient will become available using the `grad` attribute.

### Gradient Accumulation
PyTorch automatically accumulates gradients. If you specify `retain_graph=True` when calling `backward`, the computational graph will be preserved, and new gradients will be added to the `grad` field. To restart computing gradients from scratch, reset the `grad` field to 0 explicitly by calling `zero_()`.

### Computational Graph
To compute gradients, PyTorch creates and maintains a **compute graph**. For each tensor with the `requires_grad` flag set to `True`, PyTorch maintains a special function called `grad_fn`, which computes the derivative of the expression according to the chain differentiation rule.

---

## 🧩 Understanding the Jacobian Matrix in PyTorch

Here, `c` is computed using the `mean` function, thus `grad_fn` points to a function called `MeanBackward`.

In most cases, we want PyTorch to compute the gradient of a **scalar function** (such as a loss function). However, if we want to compute the gradient of a tensor with respect to another tensor, PyTorch allows us to compute the product of a **Jacobian matrix** and a given vector.

### Jacobian Definition
Suppose we have a vector function **𝑦̅ = 𝑓(𝑥̅)**, where **𝑥̅ = (𝑥₁, ..., 𝑥ₙ)** and **𝑦̅ = (𝑦₁, ..., 𝑦ₘ)**. Then, the gradient of **𝑦̅** with respect to **𝑥̅** is defined by a **Jacobian matrix**:

\[
J = 
\begin{bmatrix}
\frac{\partial y₁}{\partial x₁} & \ldots & \frac{\partial y₁}{\partial xₙ} \\
\vdots & \ddots & \vdots \\
\frac{\partial yₘ}{\partial x₁} & \ldots & \frac{\partial yₘ}{\partial xₙ}
\end{bmatrix}
\]

### Computing the Jacobian Product
Instead of giving us access to the entire Jacobian matrix, PyTorch computes the product **𝑣ᵀ · 𝐽** with a vector **𝑣 = (𝑣₁, ..., 𝑣ₘ)**. To do that, we need to call `backward()` and pass **𝑣** as an argument. The size of **𝑣** should match the size of the original tensor, with respect to which we compute the gradient.

## 🛠 Example 0: Optimization Using Gradient Descent

Let's try to use automatic differentiation to find a minimum of a simple two-variable function:

\[
f(x₁, x₂) = (x₁ - 3)² + (x₂ + 2)²
\]

Let tensor **𝑥** hold the current coordinates of a point. We start with a starting point **𝑥⁽⁰⁾ = (0, 0)** and compute the next point in a sequence using the gradient descent formula:

\[
x^{(n+1)} = x^{(n)} - \eta \nabla f
\]

Here, **η** is the so-called **learning rate** (denoted as `lr` in the code), and **∇𝑓 = \left(\frac{\partial f}{\partial x₁}, \frac{\partial f}{\partial x₂}\right)** is the gradient of **𝑓**.

---
## 📉 Linear Regression

Linear regression is defined by a straight line \( f_{W, b}(x) = Wx + b \), where \( W, b \) are model parameters that we need to find. 

An error on our dataset \( \{x_i, y_i\}_{i=1}^{N} \) (also called the **loss function**) can be defined as the mean square error:

\[
\mathcal{L}(W, b) = \frac{1}{N} \sum_{i=1}^{N} (f_{W, b}(x_i) - y_i)^2
\]

We will train the model on a series of minibatches. We will use gradient descent, adjusting model parameters using the following formulae:

\[
W^{(n+1)} = W^{(n)} - \eta \frac{\partial \mathcal{L}}{\partial W}
\]

\[
b^{(n+1)} = b^{(n)} - \eta \frac{\partial \mathcal{L}}{\partial b}
\]
---
## ⚡ Computations on GPU

To use GPU for computations, PyTorch supports moving tensors to GPU and building computational graphs for GPU. Traditionally, at the beginning of our code, we define the available computation device as `device` (which is either `cpu` or `cuda`), and then move all tensors to this device using a call like `.to(device)`.

We can also create tensors on the specified device upfront by passing the parameter `device=...` to the tensor creation code. This approach ensures that the code works seamlessly on both CPU and GPU without modification.
---
## 🗂 Example 2: Classification

Now we will consider a **binary classification problem**. A good example of such a problem would be a tumor classification between malignant and benign based on its size and age.

The core model is similar to regression, but we need to use a different **loss function**. Let's start by generating some sample data.
---
## 🧠 Training One-Layer Perceptron

Let's use **PyTorch** gradient computing machinery to train a one-layer perceptron.

Our neural network will have **2 inputs** and **1 output**. The weight matrix \( W \) will have a size of \( 2 \times 1 \), and the bias vector \( b \) will have a size of \( 1 \).

To make our code more structured, let's group all parameters into a single class.
---
### ⚙️ Notes on Updating Weights

Note that we use `W.data.zero_()` instead of `W.zero_()`. We need to do this because we cannot directly modify a tensor that is being tracked using the **Autograd** mechanism.

---

## 🧩 Logistic Loss in Classification

The core model will be the same as in the previous example, but the **loss function** will be a logistic loss. To apply logistic loss, we need to get the value of **probability** as the output of our network, i.e., we need to bring the output \( z \) to the range [0,1] using the **sigmoid activation function**: \( p = \sigma(z) \).

If we get the probability \( p_i \) for the i-th input value corresponding to the actual class \( y_i \in \{0, 1\} \), we compute the loss as:

\[
\mathcal{L}_i = - (y_i \log p_i + (1 - y_i) \log(1 - p_i))
\]

In **PyTorch**, both these steps (applying sigmoid and then logistic loss) can be done using one call to the `binary_cross_entropy_with_logits` function. Since we are training our network in minibatches, we need to average out the loss across all elements of a minibatch — and this is also done automatically by `binary_cross_entropy_with_logits`.

> The call to `binary_crossentropy_with_logits` is equivalent to a call to `sigmoid`, followed by a call to `binary_crossentropy`.
---
## 📊 Managing Datasets with PyTorch

To loop through our data, we will use the built-in **PyTorch** mechanism for managing datasets. It is based on two concepts:

- **Dataset**: The main source of data, which can be either **Iterable** or **Map-style**.
- **Dataloader**: Responsible for loading data from a dataset and splitting it into minibatches.

In our case, we will define a dataset based on a tensor and split it into minibatches of **16 elements**. Each minibatch contains two tensors:
- Input data (size = 16x2)
- Labels (a vector of length 16 of integer type representing the class number).

## 🧩 Neural Networks and Optimizers

In **PyTorch**, a special module `torch.nn.Module` is defined to represent a neural network. There are two methods to define your own neural network:

- **Sequential**, where you specify a list of layers that comprise your network.
- As a **class** inherited from `torch.nn.Module`.

The first method allows you to specify standard networks with sequential composition of layers, while the second method is more flexible and gives an opportunity to express networks of arbitrary complex architectures.

### Layers in PyTorch
Inside modules, you can use standard **layers**, such as:
- **Linear** - dense linear layer, equivalent to a one-layered perceptron. It has the same architecture as defined for our network.
- **Softmax**, **Sigmoid**, **ReLU** - layers that correspond to activation functions.
- There are also other layers for special network types, such as convolutional, recurrent, etc. These will be covered later.

> Most of the activation functions and loss functions in PyTorch are available in two forms: 
> - As a **function** (inside `torch.nn.functional` namespace).
> - As a **layer** (inside `torch.nn` namespace).  
> For activation functions, it is often easier to use functional elements from `torch.nn.functional`, without creating separate layer objects.

If we want to train a one-layer perceptron, we can just use one built-in **Linear** layer.

### 🔧 Using Parameters and Optimizers in PyTorch

As you can see, the `parameters()` method returns all the parameters that need to be adjusted during training. These correspond to the weight matrix \( W \) and bias \( b \). Note that they have `requires_grad` set to `True`, as we need to compute gradients with respect to these parameters.

PyTorch also contains built-in **optimizers**, which implement optimization methods such as **gradient descent**. Here is how we can define a **stochastic gradient descent optimizer**:

## 🔄 Defining Network as a Sequence of Layers

Now let's train a **multi-layered perceptron**. It can be defined by specifying a sequence of layers. The resulting object will automatically inherit from `Module`, meaning it will also have the `parameters` method that returns all parameters of the entire network.

## 🖥️ Defining a Network as a Class

Using a class inherited from `torch.nn.Module` is a more flexible method because we can define any computations inside it. 

The `Module` class automates many things, for example:
- It automatically understands all internal variables that are PyTorch layers.
- It gathers their parameters for optimization.

You only need to define all layers of the network as members of the class.

## ⚡ Defining a Network as PyTorch Lightning Module

Let's wrap our existing PyTorch model code into a **PyTorch Lightning** module. This allows us to work with our model more conveniently and flexibly using various Lightning methods for training and accuracy testing.

### 📦 Installing PyTorch Lightning
First, we need to install and import PyTorch Lightning. This can be done with the command:

## 📌 Takeaways

- **PyTorch** allows you to operate on tensors at a low level, providing maximum flexibility.
- There are convenient tools to work with data, such as **Datasets** and **Dataloaders**.
- You can define neural network architectures using **Sequential** syntax or by inheriting a class from `torch.nn.Module`.
- For an even simpler approach to defining and training a network, consider using **PyTorch Lightning**.




