from poortorch import poortorch
import random

def test_creation():
    print("Testing creation...")
    t1 = poortorch.zeros((2, 3))
    print(f"zeros(2, 3): {t1}")
    
    t2 = poortorch.ones((2, 3))
    print(f"ones(2, 3): {t2}")
    
    t3 = poortorch.arange(0, 5)
    print(f"arange(0, 5): {t3}")
    
    t4 = poortorch.randn((2, 2))
    print(f"randn(2, 2): {t4}")

def test_arithmetic():
    print("\nTesting arithmetic...")
    a = poortorch.ones((2, 2))
    b = poortorch.ones((2, 2))
    
    print(f"a + b: {a + b}")
    print(f"a - b: {a - b}")
    print(f"a * b: {a * b}")
    print(f"a / b: {a / b}")
    
    # Scalar ops
    print(f"a + 1: {a + 1}")

def test_matmul():
    print("\nTesting matmul...")
    a = poortorch.arange(0, 4).reshape((2, 2))
    b = poortorch.ones((2, 2))
    
    print(f"a: {a}")
    print(f"b: {b}")
    print(f"a @ b: {a @ b}")

def test_error():
    print("\nTesting error...")
    try:
        a = poortorch.zeros((2, 2))
        b = poortorch.zeros((3, 3))
        c = a + b
    except Exception as e:
        print(f"Caught expected error: {e}")

if __name__ == "__main__":
    test_creation()
    test_arithmetic()
    test_matmul()
    test_error()
