import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import random
import pytorch_lightning as pl

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

#Example 2: Classification
np.random.seed(0)
n = 100

X, Y = make_classification(n_samples=n, n_features=2, n_redundant=0, n_informative=2, flip_y=0.1, class_sep=1.5)
X = X.astype(np.float32)
Y = Y.astype(np.int32)

split = [70*n // 100, (15+70)*n // 100]
train_x, valid_x, test_x = np.split(X, split)
train_labels, valid_labels, test_labels = np.split(Y, split)

def plot_dataset(features, labels, W=None, b=None):
    fig, ax = plt.subplots(1,1)
    ax.set_xlabel('$x_i[0]$ -- (feature 1)')
    ax.set_ylabel('$x_i[1]$ -- (feature 2)')
    colors = ['r' if l else 'b' for l in labels]
    ax.scatter(features[:,0], features[:,1], marker='o', c=colors, s=100, alpha=0.5)
    if W is not None:
        min_x = min(features[:,0])
        max_x = max(features[:,1])
        min_y = min(features[:,1])*(1-.1)
        max_y = max(features[:,1])*(1+.1)
        cx = np.array([min_x, max_x], dtype=np.float32)
        cy = (0.5-W[0]*cx-b) / W[1]
        ax.plot(cx,cy,'g')
        ax.set_ylim(min_y, max_y)
    plt.show()
plot_dataset(train_x, train_labels)

#Training One-Layer Perceptron
class Network():
    def __init__(self):
        self.W = torch.randn(size=(2,1), requires_grad=True)
        self.b = torch.zeros(size=(1,), requires_grad=True)
        
    def forward(self, x):
        return torch.matmul(x, self.W)+self.b
    
    def zero_grad(self):
        self.W.data.zero_()
        self.b.data.zero_()
        
    def update(self, lr=0.1):
        self.W.data.sub_(lr*self.W.grad)
        self.b.data.sub_(lr*self.b)
        
net = Network()

def train_on_batch(net, x, y):
    z = net.forward(x).flatten()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(input=z, target=y)
    net.zero_grad()
    loss.backward()
    net.update()
    return loss

dataset = torch.utils.data.TensorDataset(torch.tensor(train_x), torch.tensor(train_labels,dtype=torch.float32))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)

batch = next(iter(dataloader))
print(batch)

for epoch in range(15):
    for (x, y) in dataloader:
        loss = train_on_batch(net, x, y)
    print('Epoch %d: last batch loss = %.4f' % (epoch, float(loss)))
    
print(net.W,net.b)

plot_dataset(train_x, train_labels, net.W.detach().numpy(),net.b.detach().numpy())

pred = torch.sigmoid(net.forward(torch.tensor(valid_x)))
torch.mean(((pred.view(-1)>0.5)==(torch.tensor(valid_labels)>0.5)).type(torch.float32))
print(pred)

#Neural Networks and Optimizers
net = torch.nn.Linear(2,1) #2inputs 1output
print(list(net.parameters()))

optim = torch.optim.SGD(net.parameters(), lr=0.1)

val_x = torch.tensor(valid_x)
val_lab = torch.tensor(valid_labels)

for ep in range(10):
    for (x,y) in dataloader:
        z = net(x).flatten()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z,y)
        optim.zero_grad()
        loss.backward()
        optim.step()
    acc = ((torch.sigmoid(net(val_x).flatten())>0.5).float()==val_lab).float().mean()
    print(f"Epoch {ep}: last batch loss = {loss}, val acc = {acc}")
    
def train(net, dataloader, val_x, val_lab, epochs=10, lr=0.5):
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    for ep in range(epochs):
        for (x,y) in dataloader:
            z = net(x).flatten()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z,y)
            optim.zero_grad()
            loss.backward()
            optim.step()
        acc = ((torch.sigmoid(net(val_x).flatten())>0.5).float()==val_lab).float().mean()
        print(f"Epoch {ep}: last batch loss = {loss}, val acc = {acc}")

net = torch.nn.Linear(2,1)
train(net, dataloader, val_x, val_lab, lr=0.03)

# define Network as a sequence of layers
net = torch.nn.Sequential(torch.nn.Linear(2,5), torch.nn.Sigmoid(), torch.nn.Linear(5,1))
print(net)

train(net, dataloader, val_x, val_lab)

#Defining a Network as a Class
class MyNet(torch.nn.Module):
    def __init__(self, hidden_size = 10, func=torch.nn.Sigmoid()):
        super().__init__()
        self.fc1 = torch.nn.Linear(2,hidden_size)
        self.func = func
        self.fc2 = torch.nn.Linear(hidden_size, 1)
        
    def forward(self,x):
        x = self.fc1(x)
        x = self.func(x)
        x = self.fc2(x)
        return x
    
net = MyNet(func=torch.nn.ReLU())
print(net)

train(net, dataloader, val_x,val_lab, lr=0.005)

#Defining a Network as PyTorch Lightning Module

class MyNetPL(pl.LightningModule):
    def __init__(self, hidden_size = 10, func = torch.nn.Sigmoid()):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, hidden_size)
        self.func = func
        self.fc2 = torch.nn.Linear(hidden_size, 1)
        
        self.val_epoch_num = 0 #logging
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.func(x)
        x = self.fc2(x)
        return x
        
    def training_step(self, batch, batch_nb):
        x, y = batchy_res = batch
        y_res = self(x).view(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_res, y)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.005)
        return optimizer
    def validation_step(self, batch, btch_nb):
        x,y = batch
        y_res = self(x).view(-1)
        val_loss = torch.nn.functional.binary_cross_entropy_with_logits(y_res, y)
        print("Epoch ", self.val_epoch_num, ": val loss = ", val_loss.item(), " val acc ", ((torch.sigmoid(y_res.flatten())>0.5).float()==y).float().mean().item(), sep = "")
        self.val_epoch_num += 1
valid_dataset = torch.utils.data.TensorDataset(torch.tensor(valid_x), torch.tensor(valid_labels, dtype=torch.float32))
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size = 16)

net = MyNetPL(func=torch.nn.ReLU())
trainer = pl.Trainer(max_epochs = 30, log_every_n_steps = 1, accelerator='gpu', devices=1)
trainer.fit(model = net, train_dataloaders= dataloader, val_dataloaders=valid_dataloader)