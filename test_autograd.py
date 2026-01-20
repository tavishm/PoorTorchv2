from poortorch import poortorch

def test_scalar_autograd():
    print("Testing scalar autograd...")
    a = poortorch.tensor(2.0, requires_grad=True, dtype=poortorch.float32)
    b = poortorch.tensor(-3.0, requires_grad=True, dtype=poortorch.float32)
    c = poortorch.tensor(10.0, requires_grad=True, dtype=poortorch.float32)
    
    e = a * b
    d = e + c
    f = poortorch.tensor(-2.0, requires_grad=True, dtype=poortorch.float32)
    L = d * f
    
    L.backward()
    
    print(f"L.data: {L}")
    print(f"a.grad: {a.grad} (expected 6.0)")
    print(f"b.grad: {b.grad} (expected -4.0)")
    print(f"c.grad: {c.grad} (expected -2.0)")
    print(f"f.grad: {f.grad} (expected 4.0)")
    
    # Assertions for float equality (approx)
    assert abs(float(a.grad) - 6.0) < 1e-5
    assert abs(float(b.grad) - (-4.0)) < 1e-5

def test_matrix_autograd():
    print("\nTesting matrix autograd...")
    # A (2x1) @ B (1x2) -> C (2x2)
    # C = [[a1*b1, a1*b2], 
    #      [a2*b1, a2*b2]]
    # L = sum(C) (implicit via backward on non-scalar) = sum(ones * C)
    # dL/dC = [[1, 1], [1, 1]]
    
    A = poortorch.tensor([[1.0], [2.0]], requires_grad=True)
    B = poortorch.tensor([[3.0, 4.0]], requires_grad=True)
    
    C = A @ B
    
    # C should be [[3, 4], [6, 8]]
    print(f"C: {C}")
    
    C.backward()
    
    print("Backprop done")
    print(f"A.grad: {A.grad}")
    print(f"B.grad: {B.grad}")
    
    # dL/dA = dL/dC @ B.T = [[1,1],[1,1]] @ [[3],[4]] = [[7], [7]]
    # dL/dB = A.T @ dL/dC = [[1, 2]] @ [[1,1],[1,1]] = [[3, 3]]
    
    # Check A.grad (shape 2x1)
    # Storage: [7, 7]
    assert abs(float(A.grad.__storage__[0]) - 7.0) < 1e-5
    assert abs(float(A.grad.__storage__[1]) - 7.0) < 1e-5
    
    # Check B.grad (shape 1x2)
    # Storage: [3, 3]
    assert abs(float(B.grad.__storage__[0]) - 3.0) < 1e-5
    assert abs(float(B.grad.__storage__[1]) - 3.0) < 1e-5

def test_types():
    print("\nTesting types...")
    a = poortorch.tensor([1, 2], dtype=poortorch.int32)
    print(f"Original: {a}")
    b = a.astype(poortorch.float32)
    print(f"Casted to float: {b}")
    assert b.dtype == poortorch.float32
    assert isinstance(b.__storage__[0], float) or isinstance(b.__storage__[0], poortorch.float32)

if __name__ == "__main__":
    test_scalar_autograd()
    test_matrix_autograd()
    test_types()
