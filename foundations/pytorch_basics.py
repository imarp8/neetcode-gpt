import torch
import torch.nn
from torchtyping import TensorType

# Round all answers to 4 decimal places: torch.round(tensor, decimals=4)
class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        # Reshape (M, N) tensor to (M*N/2, 2)
        try:
            reshaped_tensor = torch.round(
                                torch.reshape(
                                to_reshape, ((to_reshape.shape[0] * to_reshape.shape[1])//2,2)
                                ), decimals = 2)
            return reshaped_tensor
        except:
            reshaped_tensor = torch.round(
                                torch.reshape(
                                    to_reshape, (-1,2)
                                )
                                )
            return reshaped_tensor
        # Use torch.reshape(tensor, new_shape)
        pass

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        # Compute column-wise mean (average across rows)
        return torch.round(
                    torch.mean(to_avg, dim=0)
                    , decimals = 4)
        # Use torch.mean(tensor, dim=0)
        pass

    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
        # Join two tensors side-by-side along dim=1
        # Use torch.cat((a, b), dim=1)
        return torch.round(
                    torch.cat((cat_one, cat_two), dim=1)
                    , decimals = 4
        )
        pass

    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
        # Compute Mean Squared Error between prediction and target
        # Use torch.nn.functional.mse_loss(prediction, target)
        return torch.round(
                    torch.nn.functional.mse_loss(prediction, target)
                    , decimals = 4
        )
        pass
