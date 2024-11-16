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
