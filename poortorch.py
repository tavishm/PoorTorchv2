import numpy as np
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

    class tensor:
        def __init__(self, xl: list):
            # Storage
            self.__storage__ = []
            self.helper._flatten_list(xl)

            # Shape
            self.dtype = poortorch.int64
            self.shape = tuple(self.helper._shape_iterable(xl)[::-1])

            # Strides
            self.stride = [None for _ in range(self.shape)]
            for i in reversed(range(len(self.shape) - 1)):
                self.stride[i] = self.shape[i + 1] * self.stride[i + 1]
            
            # Unifying dtype
            for i in range(len(self.__storage__)): self.__storage__[i] == self.dtype(self.__storage__[i])


        
        class helper:
            def _flatten_list(self, xl: list):
                if isinstance(xl, (int, float)):
                    self.__storage__.append(xl)
                elif isinstance(xl, list):
                    for sub_xl in xl:
                        self.helper._flatten_list(sub_xl)
                else:
                    raise Exception("Tensor can only contain int, float or lists 😔")
                
            def _shape_iterable(self, xl: list):
                if not isinstance(xl, list):
                    raise Exception("Given list does not have a definite shape 😔")

                elif all(isinstance(i, (int, float)) for i in xl):
                    for i in xl:
                        if isinstance(i, float): #TODO: Better datatype management
                            self.dtype = poortorch.float32
                    
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