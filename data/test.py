from random import sample

# test = [1,2,3,4,5]

# print("sample(test, 3) = ", sample(test, 3))

import torch

x = torch.tensor([1.0,2.0,3.0], requires_grad=True)
y = x.clone()
y[0] = 10
print(x)
print(y)

z = x.clone().detach()
z[0] = 20
print(x)
print(y)
print(z)


