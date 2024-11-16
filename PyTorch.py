import torch
torch.__version__

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
    