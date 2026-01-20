# PoorTorch v2 😔

The most inefficient deep learning framework ever created. Runs purely on Python lists and CPU.

## Features
- **Tensors**: Inefficiently stored as Python lists.
- **Autograd**: Full backpropagation engine.
- **Operations**: `+`, `-`, `*`, `/`, `@` (matmul), `reshape`, `transpose`.
- **Dtypes**: Supports standard types (`float32`, `int64`, etc.).

## Usage
```python
from poortorch import poortorch

# Create tensors with autograd
x = poortorch.tensor([2.0], requires_grad=True)
y = poortorch.tensor([3.0], requires_grad=True)

# Build graph
z = x * y + 10

# Backpropagate
z.backward()

print(f"z: {z}")       # 16.0
print(f"x.grad: {x.grad}") # 3.0 (dz/dx = y)
```
**Warning**: Do not use in production. 😔

## Disclaimer
This project is a programming exercise for educational purposes.

## Contact
Created by Tavish. Reach out for any questions!
tavish.mankash@gmail.com
