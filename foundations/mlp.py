import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        
        a=x

        for i in range(len(weights)):
            z = a @ weights[i] + biases[i]

            # Applying Relu
            if i == len(weights) - 1:
                return np.round(z, 5)
            else:
                a = np.maximum(0,z)


        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        pass
