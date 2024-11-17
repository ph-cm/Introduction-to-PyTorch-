import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import random

#Simple Tensors
a = torch.tensor([[1,2],[3,4]])
print(a)
a = torch.randn(size=(10,3))
print(a)

print(a-a[0])
print(torch.exp(a)[0].numpy())

#In-place and ou-place operations
u = torch.tensor(5)
print("Result when adding out-of-place: ", u.add(torch.tensor(3)))
u.add_(torch.tensor(3))
print("Result after adding in-place: ", u)

s = torch.zeros_like(a[0])
for i in a:
    s.add_(i)
    
print(s)

torch.sum(a,axis=0)

#Computing Gradients
a = torch.randn(size = (2,2), requires_grad=True)
b = torch.randn(size=(2,2))

c = torch.mean(torch.sqrt(torch.square(a) + torch.square(b)))
c.backward() #compute all gradients
print(a.grad) #the gradient of 'c' with respect to 'a'

c = torch.mean(torch.sqrt(torch.square(a) + torch.square(b)))
c.backward(retain_graph=True)
c.backward(retain_graph=True)
print(a.grad)
a.grad.zero_()
c.backward()
print(a.grad)

print(c)

c = torch.sqrt(torch.square(a) + torch.square(b))
c.backward(torch.eye(2)) #eye == 2x2 identity matrix
print(a.grad)

#Example: Optimization Using  Gradient Descent
x = torch.zeros(2,requires_grad=True)
f = lambda x : (x-torch.tensor([3,-2])).pow(2).sum()
lr = 0.1

for i in range(15):
    y = f(x)
    y.backward()
    gr = x.grad
    x.data.add_(-lr*gr)
    x.grad.zero_()
    print("Step {}: x[0]={}, x[1]={}".format(i,x[0],x[1]))
    
#Example: Linear Regression

np.random.seed(13)
train_x = np.linspace(0, 3, 120)
train_labels = 2 * train_x + 0.9 + np.random.randn(*train_x.shape) * 0.5

plt.scatter(train_x, train_labels)
plt.show()

input_dim = 1
output_dim = 1
learning_rate = 0.1

#This is our weight matrix
w = torch.tensor([100.0], requires_grad=True, dtype=torch.float32)
# This is our Bias vector
b = torch.zeros(size=(output_dim,), requires_grad=True)

def f(x):
    return torch.matmul(x,w) + b

def compute_loss(labels, predictions):
    return torch.mean(torch.square(labels - predictions))

def train_on_batch(x,y):
    predictions = f(x)
    loss = compute_loss(y, predictions)
    loss.backward()
    w.data.sub_(learning_rate * w.grad)
    b.data.sub_(learning_rate * b.grad)
    w.grad.zero_()
    b.grad.zero_()
    return loss

#shuffe data
indices = np.random.permutation(len(train_x))
features = torch.tensor(train_x[indices], dtype=torch.float32)
labels = torch.tensor(train_labels[indices],dtype=torch.float32)

batch_size = 4
for epoch in range(10):
    for i in range(0,len(features), batch_size):
        loss = train_on_batch(features[i:i+batch_size].view(-1,1), labels[i:i+batch_size])
    print('Epoch %d: last batch loss = %.4f' % (epoch, float(loss)))

w,b

plt.scatter(train_x, train_labels)
x = np.array([min(train_x), max(train_x)])
with torch.no_grad():
    y = w.numpy()*x+b.numpy()
plt.plot(x,y,color='red')
plt.show()
    
#Computation on GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
 
print('Doing computations on ' + device)

# Indicates devices
w = torch.tensor([100.0], requires_grad=True, dtype = torch.float32, device=device)
b = torch.zeros(size=(output_dim,), requires_grad=True, device=device)

def f(x):
    return torch.matmul(x,w) + b
def compute_loss(labels, predictions):
    return torch.mean(torch.square(labels - predictions))

def train_on_batch(x, y):
    predictions = f(x)
    loss = compute_loss(y, predictions)
    loss.backward()
    w.data.sub_(learning_rate * w.grad)
    b.data.sub_(learning_rate * b.grad)
    w.grad.zero_()
    b.grad.zero_()
    return loss
batch_size = 4
for epoch in range(10):
    for i in range(0, len(features), batch_size):
        #move data to required device
        loss = train_on_batch(features[i:i+batch_size].view(-1,1).to(device), labels[i:i+batch_size].to(device))
    print('Epoch %d: last batch loss = %.4f' % (epoch, float(loss)))
    