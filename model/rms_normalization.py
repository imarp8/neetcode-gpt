import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:

        x = np.array(x)
        gamma = np.array(gamma)

        rms = np.sqrt(np.mean(np.square(x)) + eps)
        x_hat = x / rms

        return np.round(x_hat * gamma,4)
        
        pass
