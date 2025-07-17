import numpy as np
import typing
from typing import List, Union, Optional
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

    class tensor:
        def __init__(self, xl: Union[list, float, int], dtype: "poortorch.dtype" = None):
            if isinstance(xl, (int, float)): # Scalars
                self.shape = []
                self.stride = []
                self.dtype = dtype if dtype else (poortorch.float32 if isinstance(xl, float) else poortorch.int64)
                self.__storage__ = self.dtype(xl)
            elif isinstance(xl, list): 
                # Storage
                self.__storage__ = []
                self.helper._flatten_list(xl)

                # Shape and Deciding dtype: If shape is None, int64 or float32 is selected depending on data. Otherwise, specified dtype is used.
                self.dtype = dtype
                if not self.dtype: 
                    self.shape = tuple(self.helper._shape_iterable(xl)[::-1]) # Sets dtype to flaot32 if float is encountered
                    if not self.dtype: self.dtype = poortorch.int64           # Sets dtype to int64 if float is not encountered
                else:
                    self.shape = tuple(self.helper._shape_iterable(xl)[::-1]) # Throws error if float is encountered in int dtypes. It's okay to encounter ints in float dtypes.

                # Strides
                self.stride = [None for _ in range(self.shape)]
                for i in reversed(range(len(self.shape) - 1)):
                    self.stride[i] = self.shape[i + 1] * self.stride[i + 1]
                
                # Converting data to decided dtype
                for i in range(len(self.__storage__)): self.__storage__[i] == poortorch.tensor(self.__storage__[i], dtype=self.dtype)
            else:
                raise Exception("Tensor can only be created from int, float or lists 😔")
            
        def __str__(self):
            if len(self.shape) == 0:
                return f"poortorch.tensor({self.__storage__}, dtype={self.dtype.__name__})"
            
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
        
        def __int__(self):
            if len(self.shape) != 0:
                raise Exception("Cannot convert non-scalar tensor with shape to int 😔")
            else:
                return int(self.__storage__)
        
        def __float__(self):
            if len(self.shape) != 0:
                raise Exception("Cannot convert non-scalar tensor with shape to float 😔")
            else:
                return float(self.__storage__)            

        
        class helper:
            def _flatten_list(self, xl: list) -> list:
                if isinstance(xl, (int, float)):
                    self.__storage__.append(xl)
                elif isinstance(xl, list):
                    for sub_xl in xl:
                        self.helper._flatten_list(sub_xl)
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
                        ds.append(poortorch.tensor._shape_iterable(k_item, self))

                    same_shape = all(ds[0] == j for j in ds)
                    if not same_shape:
                        raise Exception("Given list does not have a definite shape 😔")
                    if same_shape:
                        shape.extend(ds[0])
                        shape.append(len(xl))

                return shape