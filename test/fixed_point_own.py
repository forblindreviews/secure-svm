import numpy as np


class FpxArray(np.ndarray):
    def __new__(cls, input_array, scaling=1):
        obj = np.asarray(input_array).view(cls)
        obj.scaling = scaling
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'scaling', None)


arr = np.array([1, 2, 3, 4, 5])
obj = FpxArray(arr, scaling=1/100)

arr2 = np.array([1, 2, 3, 4, 5])
obj2 = FpxArray(arr, scaling=1/100)

print(obj + obj2)