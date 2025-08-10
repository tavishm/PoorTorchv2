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

    class tensor:
        def __init__(self, xl: Union[list, float, int], dtype: "poortorch.dtype" = None, manual_creation_dict: dict = None):
            if isinstance(xl, (int, float)): # Scalars
                self.shape = []
                self.stride = []
                self.dtype = dtype if dtype else (poortorch.float32 if isinstance(xl, float) else poortorch.int64)
                self.__storage__ = self.dtype(xl)
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
                self.stride = [None for _ in range(len(self.shape))]
                self.stride[-1] = 1
                for i in reversed(range(len(self.shape) - 1)):
                    self.stride[i] = self.shape[i + 1] * self.stride[i + 1]
                
                # Converting data to decided dtype
                for i in range(len(self.__storage__)): self.__storage__[i] == poortorch.tensor(self.__storage__[i], dtype=self.dtype)
            else:
                raise Exception("Tensor can only be created from int, float or lists 😔")
            
        def __str__(self) -> str:
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
        
        def __int__(self) -> int:
            if len(self.shape) != 0:
                raise Exception("Cannot convert non-scalar tensor with shape to int 😔")
            else:
                return int(self.__storage__)
        
        def __float__(self) -> float:
            if len(self.shape) != 0:
                raise Exception("Cannot convert non-scalar tensor with shape to float 😔")
            else:
                return float(self.__storage__)
        
        def __getitem__(self, idx) -> 'poortorch.tensor':
            
            shape= self.shape
            dat= self.__storage__
            stride= self.stride
    
            #formatting the idx variable
            if isinstance(idx, slice):
                idx=[idx,]
            #single element case
            if all(isinstance(i,int) for i in idx) and len(idx)==len(shape):
                loc=0
                for i in zip(idx,stride):
                    loc+=i[0]*i[1]
                return poortorch.tensor(dat[loc])
                
            idx=list(idx)
            for i in range(len(idx)):
                if isinstance(idx[i], int):
                    idx[i]=slice(idx[i],idx[i]+1,1)

            #exception handling
            if len(idx)>len(shape):
                raise Exception('Number of parameters exceeded order of tensor 😔')
            for i in idx:
                if not (isinstance(i.start,(int, type(None))) and isinstance(i.stop,(int, type(None))) and isinstance(i.step,(int, type(None)))) :
                    raise Exception("indices must be integers 😔")
            for i in zip(idx, shape[:len(idx)]):
                if not ((i[0].start==None or 0<=i[0].start<=i[1]) and (i[0].start==None or 0<=i[0].stop<=i[1]) and (i[0].step==None or 0<=i[0].step<=i[1])):
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
            return poortorch.tensor(l)

        
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
                if all(isinstance(dim, int) and dim > 0 for dim in shape): raise Exception("Shape must be a tuple of positive integers 😔")
                __storage__ = [value_function() for _ in range(math.prod(shape))]
                if not dtype:
                    dtype = poortorch.float32 if isinstance(__storage__[0], float) else poortorch.int64
                
                manual_creation_dict = {
                    "__storage__": __storage__,
                    "dtype": dtype,
                    "shape": shape,
                }

                return poortorch.tensor([], None, manual_creation_dict)
            
a= poortorch.tensor([[[1,2,3,4,5],[6,7,8,9,10]],[[11,12,13,14,15],[16,17,18,19,20]]])
b= poortorch.tensor([[1,2,3,4,5],[6,7,8,9,0]])
c= poortorch.tensor([1,2,3,4,5])
