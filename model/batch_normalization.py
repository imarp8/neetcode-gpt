import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists

        x = np.array(x)
        gamma = np.array(gamma)
        beta = np.array(beta)

        running_mean = np.array(running_mean)
        running_var = np.array(running_var)



        if training:
            # # calculate the fearture wise mean and std
            # feature_mean = []
            # feature_std = []

            # for n in range(x.shape[1]):
            #     f = []
            #     for m in range(x.shape[0]):
            #         f.append(x[m][n])
            #     feature_mean.append(np.mean(f))
            #     feature_std.append(np.std(f))

            # feature_mean = np.array(feature_mean)
            # feature_std = np.array(feature_std)

            feature_mean = np.mean(x, axis = 0)
            feature_std = np.std(x, axis = 0)

            # Normalise each row:
            norm_row = []
            for i in range(x.shape[0]):
                row = (x[i] - feature_mean)/np.sqrt(feature_std**2 + eps)
                norm_row.append(row)
        
            x_hat = np.array(norm_row)

            # Running mean and running variance:
            running_mean = (1 - momentum) * running_mean + momentum * feature_mean
            running_var = (1 - momentum) * running_var + momentum * (feature_std**2)
        else:
            x_hat = (x - running_mean) / np.sqrt( running_var + eps)


        # Affine Transformation:
        y = x_hat * gamma + beta

        y = np.round(y, 4)
        running_mean = np.round(running_mean, 4)
        running_var = np.round(running_var, 4)

        return (
            y.tolist(),
            running_mean.tolist(),
            running_var.tolist()
        )
        pass
