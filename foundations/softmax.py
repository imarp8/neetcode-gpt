import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        result = np.array([np.round(1/(1+np.exp(-i)),5) for i in z],
                        dtype = np.float64)
        return result
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        pass

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        result = np.array([i if i> 0 else 0 for i in z],
                        dtype = np.float64)
        # Formula: max(0, z) element-wise
        return result
        pass
