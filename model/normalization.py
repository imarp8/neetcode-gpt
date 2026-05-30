import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        eps = 1e-5
        var = np.std(x)**2
        x_hat = (x - np.mean(x)) / np.sqrt(var + eps)

        """Keep every token's internal representation well-behaved so information can flow through 
        dozens or hundreds of layers without exploding or vanishing."""
        scale_shift_x = gamma * x_hat + beta
        return np.round(scale_shift_x, 5)
        pass
