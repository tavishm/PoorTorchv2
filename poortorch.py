import numpy as np
import typing
from typing import List, Union, Optional, Callable
from typing import TypeVar, Generic
import math


class poortorch:
    # dtype
    class int8(np.int8): pass
    class int16(np.int16): pass
    class int32(np.int32): pass
    class int64(np.int64): pass

    class float16(np.float16): pass
    class float32(np.float32): pass
    class float64(np.float64): pass

    dtype = TypeVar("dtype", "poortorch.int8", "poortorch.int16", "poortorch.int32", "poortorch.int64",
                           "poortorch.float16", "poortorch.float32", "poortorch.float64")
    
    # Tensor Creation
    def zeros(shape: tuple[int], dtype: "poortorch.dtype" = None) -> "poortorch.tensor":
        return poortorch.tensor.helper._create_per_value_independent_tensor(shape, lambda: 0, dtype)

    def ones(shape: tuple[int], dtype: "poortorch.dtype" = None) -> "poortorch.tensor":
        return poortorch.tensor.helper._create_per_value_independent_tensor(shape, lambda: 1, dtype)
    
    def randn(shape: tuple[int], dtype: "poortorch.dtype" = None) -> "poortorch.tensor":
        import random
        return poortorch.tensor.helper._create_per_value_independent_tensor(shape, lambda: random.gauss(0, 1), dtype)

    def arange(start: int, end: int = None, step: int = 1, dtype: "poortorch.dtype" = None) -> "poortorch.tensor":
        if end is None:
            end = start
            start = 0
            
        data = list(range(start, end, step))
        return poortorch.tensor(data, dtype=dtype)

    class tensor:
        def __init__(self, xl: Union[list, float, int], dtype: "poortorch.dtype" = None, requires_grad: bool = False, _children: tuple = (), _op: str = '', manual_creation_dict: dict = None):
            self.grad = None
            self._backward = lambda: None
            self._prev = set(_children)
            self._op = _op
            self.requires_grad = requires_grad

            if isinstance(xl, (int, float)): # Scalars
                self.shape = ()
                self.stride = []
                self.dtype = dtype if dtype else (poortorch.float32 if isinstance(xl, float) else poortorch.int64)
                self.__storage__ = [self.dtype(xl)]
            elif isinstance(xl, list): 
                if manual_creation_dict: 
                    # If a tensor is created outside of the list to tensor implementation, only a flat list, shape and dtype are required. 
                    # Stride is calculated automatically. __storage__ must be list[int, float]. They are converted to the specified dtype.
                    if dtype or xl: raise Exception("dtype must be None and input list provided must be [] if using manual dict creation 😔")
                    self.__storage__ = manual_creation_dict["__storage__"]
                    self.dtype = manual_creation_dict["dtype"]
                    self.shape = manual_creation_dict["shape"]
                else:
                    # Storage
                    self.__storage__ = []
                    poortorch.tensor.helper._flatten_list(self, xl)

                    # Shape and Deciding dtype: If shape is None, int64 or float32 is selected depending on data. Otherwise, specified dtype is used.
                    self.dtype = dtype
                    if not self.dtype: 
                        self.shape = tuple(poortorch.tensor.helper._shape_iterable(self, xl)[::-1]) # Sets dtype to flaot32 if float is encountered
                        if not self.dtype: self.dtype = poortorch.int64           # Sets dtype to int64 if float is not encountered
                    else:
                        self.shape = tuple(poortorch.tensor.helper._shape_iterable(self, xl)[::-1]) # Throws error if float is encountered in int dtypes. It's okay to encounter ints in float dtypes.

                # Strides
                if len(self.shape) > 0:
                    self.stride = [None for _ in range(len(self.shape))]
                    self.stride[-1] = 1
                    for i in reversed(range(len(self.shape) - 1)):
                        self.stride[i] = self.shape[i + 1] * self.stride[i + 1]
                else:
                    self.stride = []
                
                # Converting data to decided dtype
                # Check if elements are already correct type if possible to avoid redundant work, but for "inefficient" we re-cast
                if not manual_creation_dict:
                     for i in range(len(self.__storage__)): self.__storage__[i] = self.dtype(self.__storage__[i])
            else:
                raise Exception("Tensor can only be created from int, float or lists 😔")

        def backward(self):
            topo = []
            visited = set()
            def build_topo(v):
                if v not in visited:
                    visited.add(v)
                    for child in v._prev:
                        build_topo(child)
                    topo.append(v)
            build_topo(self)

            self.grad = poortorch.ones(self.shape, dtype=self.dtype) # Implicitly gradient of self wrt self is 1s
            
            for node in reversed(topo):
                node._backward()

        def zero_grad(self):
            self.grad = None

        def astype(self, dtype: "poortorch.dtype") -> "poortorch.tensor":
             manual_dict = {
                "__storage__": [dtype(x) for x in self.__storage__],
                "dtype": dtype,
                "shape": self.shape
            }
             return poortorch.tensor([], None, manual_creation_dict=manual_dict, requires_grad=self.requires_grad)

        def to(self, dtype: "poortorch.dtype") -> "poortorch.tensor":
             return self.astype(dtype) # Alias for pytorch users
            
        def __str__(self) -> str:
            if len(self.shape) == 0:
                return f"poortorch.tensor({self.__storage__[0]}, dtype={self.dtype.__name__})"
            
            def format_tensor(data, shape, offset=0, depth=0):
                if depth == len(shape) - 1:
                    # Last dimension - print actual values
                    start_idx = offset
                    end_idx = offset + shape[depth]
                    values = data[start_idx:end_idx]
                    
                    # Clip if too many elements
                    if len(values) > 6:
                        formatted = [str(values[i]) for i in range(3)]
                        formatted.append('...')
                        formatted.extend([str(values[i]) for i in range(-3, 0)])
                        return '[' + ', '.join(formatted) + ']'
                    else:
                        return '[' + ', '.join(map(str, values)) + ']'
                else:
                    # Higher dimensions - recurse
                    elements = []
                    stride = 1
                    for i in range(depth + 1, len(shape)):
                        stride *= shape[i]
                    
                    dim_size = shape[depth]
                    if dim_size > 6:
                        # Show first 3 and last 3 elements
                        for i in range(3):
                            elements.append(format_tensor(data, shape, offset + i * stride, depth + 1))
                        elements.append('...')
                        for i in range(dim_size - 3, dim_size):
                            elements.append(format_tensor(data, shape, offset + i * stride, depth + 1))
                    else:
                        for i in range(dim_size):
                            elements.append(format_tensor(data, shape, offset + i * stride, depth + 1))
                    
                    if depth == 0:
                        return '[' + ',\n '.join(elements) + ']'
                    else:
                        return '[' + ', '.join(elements) + ']'
            
            formatted_data = format_tensor(self.__storage__, self.shape)
            return f"poortorch.tensor({formatted_data}, dtype={self.dtype.__name__})"
        
        def __int__(self) -> int:
            if len(self.shape) != 0:
                raise Exception("Cannot convert non-scalar tensor with shape to int 😔")
            else:
                return int(self.__storage__[0])
        
        def __float__(self) -> float:
            if len(self.shape) != 0:
                raise Exception("Cannot convert non-scalar tensor with shape to float 😔")
            else:
                return float(self.__storage__[0])
        
        def __getitem__(self, idx) -> 'poortorch.tensor':
            
            shape= self.shape
            dat= self.__storage__
            if isinstance(idx, slice):
                idx=(idx,)
            
            #exception handling
            if len(idx)>len(shape):
                raise Exception('Number of parameters exceeded order of tensor 😔')
            for i in idx:
                if not (isinstance(i.start,(int, type(None))) and isinstance(i.stop,(int, type(None))) and isinstance(i.step,(int, type(None)))) :
                    raise Exception("indices must be integers 😔")
            for i in zip(idx, shape[:len(idx)]):
                if not ((i[0].start==None or 0<=i[0].start<=i[1]) and (i[0].start==None or 0<=i[0].stop<=i[1]) and (i[0].start==None or 0<=i[0].step<=i[1])):
                    raise Exception('index out of range 😔')
            
            #converting tuple of slice objects into a nested list and replacing 'None's 
            idxl=[]
            for i in range(len(shape)):
                idxl.append([0,shape[i],1])
            for i in range(len(idx)):
                if idx[i].start!=None:
                    idxl[i][0]= idx[i].start

                if idx[i].stop!=None:
                    idxl[i][1]=idx[i].stop
                
                if idx[i].step==None:
                    idxl[i][2]=1
                else: 
                    idxl[i][2]=idx[i].step  
            
            l=[]
            def get(dat,shape,idxl):
                if len(idxl)==1:
                    l.append(dat[slice(*idxl[0])])
                else:
                    for i in range(idxl[0][0],idxl[0][1],idxl[0][2]):
                        get(dat[i*math.prod(shape[1:]):(i+1)*math.prod(shape[1:])], shape[1:],idxl[1:])
                    
            get(dat, shape, idxl)
            return poortorch.tensor(l[0])

        def reshape(self, new_shape: tuple[int]) -> 'poortorch.tensor':
            if math.prod(new_shape) != math.prod(self.shape):
                raise Exception(f"Cannot reshape tensor of size {math.prod(self.shape)} to {new_shape} 😔")
            
            # Create manual creation dict to reuse storage
            manual_dict = {
                "__storage__": list(self.__storage__), # Copy storage
                "dtype": self.dtype,
                "shape": new_shape
            }
            return poortorch.tensor([], None, manual_creation_dict=manual_dict)

        def _accumulate_grad(self, grad):
            if not self.requires_grad: return
            if self.grad is None:
                 self.grad = poortorch.zeros(self.shape, dtype=self.dtype)
            
            # Simple accumulation, assume grad matches shape (checked in ops)
            # Naive loop for accumulation to match style if we wanted, but let's assume
            # we can use the __add__ of tensors properly, but grad += grad is an inplace op?
            # poortorch tensors are creating new tensors on add. 
            # So: self.grad = self.grad + grad
            if isinstance(self.grad, poortorch.tensor):
                 self.grad = self.grad + grad
            else:
                 # Should not happen if initialized correctly
                 self.grad = grad

        def __add__(self, other) -> 'poortorch.tensor':
            other = other if isinstance(other, poortorch.tensor) else poortorch.tensor(other, dtype=self.dtype)
            
            if self.shape != other.shape:
                raise Exception(f"Shape mismatch: {self.shape} and {other.shape} 😔")

            # Forward pass
            new_storage = []
            for i in range(len(self.__storage__)):
                new_storage.append(self.__storage__[i] + other.__storage__[i])
            
            manual_dict = {
                "__storage__": new_storage,
                "dtype": self.dtype,
                "shape": self.shape
            }
            out = poortorch.tensor([], None, manual_creation_dict=manual_dict, 
                                   requires_grad=self.requires_grad or other.requires_grad,
                                   _children=(self, other), _op='+')
            
            def _backward():
                self._accumulate_grad(out.grad)
                other._accumulate_grad(out.grad)
            out._backward = _backward
            
            return out

        def __sub__(self, other) -> 'poortorch.tensor':
            other = other if isinstance(other, poortorch.tensor) else poortorch.tensor(other, dtype=self.dtype)
            
            if self.shape != other.shape:
                raise Exception(f"Shape mismatch: {self.shape} and {other.shape} 😔")

            # Forward pass
            new_storage = []
            for i in range(len(self.__storage__)):
                new_storage.append(self.__storage__[i] - other.__storage__[i])
            
            manual_dict = {
                "__storage__": new_storage,
                "dtype": self.dtype,
                "shape": self.shape
            }
            out = poortorch.tensor([], None, manual_creation_dict=manual_dict,
                                   requires_grad=self.requires_grad or other.requires_grad,
                                   _children=(self, other), _op='-')

            def _backward():
                self._accumulate_grad(out.grad)
                # other.grad += -out.grad -> other.grad += out.grad * -1
                other._accumulate_grad(out.grad * -1)
            out._backward = _backward
            
            return out
            
        def __mul__(self, other) -> 'poortorch.tensor':
            other = other if isinstance(other, poortorch.tensor) else poortorch.tensor(other, dtype=self.dtype)
            
            if self.shape != other.shape:
                raise Exception(f"Shape mismatch: {self.shape} and {other.shape} 😔")

            # Forward pass
            new_storage = []
            for i in range(len(self.__storage__)):
                new_storage.append(self.__storage__[i] * other.__storage__[i])
                
            manual_dict = {
                "__storage__": new_storage,
                "dtype": self.dtype,
                "shape": self.shape
            }
            out = poortorch.tensor([], None, manual_creation_dict=manual_dict,
                                   requires_grad=self.requires_grad or other.requires_grad,
                                   _children=(self, other), _op='*')

            def _backward():
                # self.grad += other * out.grad
                self._accumulate_grad(other * out.grad)
                # other.grad += self * out.grad
                other._accumulate_grad(self * out.grad)
            out._backward = _backward
            
            return out
            
        def __truediv__(self, other) -> 'poortorch.tensor':
            other = other if isinstance(other, poortorch.tensor) else poortorch.tensor(other, dtype=self.dtype)

            if self.shape != other.shape:
                raise Exception(f"Shape mismatch: {self.shape} and {other.shape} 😔")

            # Forward pass
            new_storage = []
            for i in range(len(self.__storage__)):
                 new_storage.append(self.__storage__[i] / other.__storage__[i])

            manual_dict = {
                "__storage__": new_storage,
                "dtype": self.dtype,
                "shape": self.shape
            }
            out = poortorch.tensor([], None, manual_creation_dict=manual_dict,
                                   requires_grad=self.requires_grad or other.requires_grad,
                                   _children=(self, other), _op='/')
            
            def _backward():
                # self.grad += out.grad * (1 / other)
                self._accumulate_grad(out.grad * (other.pow(-1)))
                # other.grad += -self * out.grad / other**2
                #            = out.grad * (-self * other**-2)
                other._accumulate_grad(out.grad * (self * -1 * other.pow(-2)))
            out._backward = _backward

            return out

        def pow(self, exponent) -> 'poortorch.tensor':
            # Needed for division backward pass
            new_storage = [x ** exponent for x in self.__storage__]
            manual_dict = {
                "__storage__": new_storage,
                "dtype": self.dtype,
                "shape": self.shape
            }
            out = poortorch.tensor([], None, manual_creation_dict=manual_dict,
                                  requires_grad=self.requires_grad,
                                  _children=(self,), _op=f'**{exponent}')
            
            def _backward():
                # d/dx (x^n) = n * x^(n-1)
                # self.grad += out.grad * n * self**(n-1)
                self._accumulate_grad(out.grad * (self.pow(exponent - 1) * exponent))
            out._backward = _backward
            return out

        def reshape(self, new_shape: tuple[int]) -> 'poortorch.tensor':
            if math.prod(new_shape) != math.prod(self.shape):
                raise Exception(f"Cannot reshape tensor of size {math.prod(self.shape)} to {new_shape} 😔")
            
            # Create manual creation dict to reuse storage
            manual_dict = {
                "__storage__": list(self.__storage__), # Copy storage
                "dtype": self.dtype,
                "shape": new_shape
            }
            out = poortorch.tensor([], None, manual_creation_dict=manual_dict, requires_grad=self.requires_grad, _children=(self,), _op='reshape')
            
            def _backward():
                self._accumulate_grad(out.grad.reshape(self.shape))
            out._backward = _backward
            
            return out

        @property
        def T(self):
            return self.transpose()

        def transpose(self) -> 'poortorch.tensor':
            # Only 2D transpose for now
            if len(self.shape) != 2: raise Exception("Transpose only supported for 2D tensors 😔")
            
            rows, cols = self.shape
            new_shape = (cols, rows)
            new_storage = [0] * (rows * cols)
            
            for i in range(rows):
                for j in range(cols):
                     # self[i, j] -> new[j, i]
                     # self flat: i * cols + j
                     # new flat: j * rows + i
                     new_storage[j * rows + i] = self.__storage__[i * cols + j]
            
            manual_dict = {
                "__storage__": new_storage,
                "dtype": self.dtype,
                "shape": new_shape
            }
            out = poortorch.tensor([], None, manual_creation_dict=manual_dict, requires_grad=self.requires_grad, _children=(self,), _op='T')
            
            def _backward():
                self._accumulate_grad(out.grad.transpose())
            out._backward = _backward

            return out

        def __matmul__(self, other) -> 'poortorch.tensor':
            if not isinstance(other, poortorch.tensor):
                raise Exception("Matmul only supports poortorch tensors 😔")
            
            # Only implementing 2D matmul for now as per "inefficient" spec, maybe simple batching if needed
            if len(self.shape) != 2 or len(other.shape) != 2:
                raise Exception("Matmul only supports 2D tensors for now 😔")
                
            if self.shape[1] != other.shape[0]:
                raise Exception(f"Shape mismatch for matmul: {self.shape} and {other.shape} 😔")
                
            rows_a = self.shape[0]
            cols_a = self.shape[1] # same as rows_b
            cols_b = other.shape[1]
            
            result_storage = [0] * (rows_a * cols_b)
            
            # Naive O(N^3) implementation
            for i in range(rows_a):
                for j in range(cols_b):
                    sum_val = 0
                    for k in range(cols_a):
                        # Calculate flat indices
                        idx_a = i * self.stride[0] + k * self.stride[1]
                        idx_b = k * other.stride[0] + j * other.stride[1]
                        sum_val += self.__storage__[idx_a] * other.__storage__[idx_b]
                    
                    # Result is row-major
                    result_idx = i * cols_b + j
                    result_storage[result_idx] = sum_val
                    
            manual_dict = {
                "__storage__": result_storage,
                "dtype": self.dtype, # Propagate dtype? or promote? keeping simple
                "shape": (rows_a, cols_b)
            }
            out = poortorch.tensor([], None, manual_creation_dict=manual_dict,
                                   requires_grad=self.requires_grad or other.requires_grad,
                                   _children=(self, other), _op='@')
            
            def _backward():
                # C = A @ B
                # dA = dC @ B.T
                self._accumulate_grad(out.grad @ other.T)
                # dB = A.T @ dC
                other._accumulate_grad(self.T @ out.grad)
            out._backward = _backward

            return out

        
        class helper:
            def _flatten_list(self, xl: list) -> list:
                if isinstance(xl, (int, float)):
                    self.__storage__.append(xl)
                elif isinstance(xl, list):
                    for sub_xl in xl:
                        poortorch.tensor.helper._flatten_list(self, sub_xl)
                else:
                    raise Exception("Tensor can only contain int, float or lists 😔")
                
            def _shape_iterable(self, xl: list) -> list:
                if not isinstance(xl, list):
                    raise Exception("Given list does not have a definite shape 😔")

                elif all(isinstance(i, (int, float)) for i in xl):
                    for i in xl:
                        if isinstance(i, float): #TODO: Better datatype management
                            if not self.dtype: 
                                self.dtype = poortorch.float32
                            if self.dtype in [poortorch.int64, poortorch.int32, poortorch.int16, poortorch.int8]:
                                raise Exception("Encountered float in int dtype tensor 😔")    
                    return [len(xl)]
                
                elif not all(isinstance(i, (int, float, list)) for i in xl):
                    raise Exception("Given list has elements other than list, int or float 😔")
                
                else:
                    shape = []
                    ds = []
                    for k_item in xl: # Changed from iterating by index to iterating by item
                        ds.append(poortorch.tensor.helper._shape_iterable(self, k_item))

                    same_shape = all(ds[0] == j for j in ds)
                    if not same_shape:
                        raise Exception("Given list does not have a definite shape 😔")
                    if same_shape:
                        shape.extend(ds[0])
                        shape.append(len(xl))

                return shape
            
            def _create_per_value_independent_tensor(shape: tuple[int], value_function: Callable, dtype: "poortorch.dtype") -> "poortorch.tensor":
                if not all(isinstance(dim, int) and dim > 0 for dim in shape): raise Exception("Shape must be a tuple of positive integers 😔")
                __storage__ = [value_function() for _ in range(math.prod(shape))]
                if not dtype:
                    dtype = poortorch.float32 if isinstance(__storage__[0], float) else poortorch.int64
                
                manual_creation_dict = {
                    "__storage__": __storage__,
                    "dtype": dtype,
                    "shape": shape,
                }

                return poortorch.tensor([], None, manual_creation_dict=manual_creation_dict)