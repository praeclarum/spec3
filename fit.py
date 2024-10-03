"""This is the code used to determine the linear fits to convert to and from SPEC4"""
import torch
from torch import Tensor, nn
from tqdm.autonotebook import tqdm

import conversions

class WavelengthsModel(nn.Module):
    """A model that predicts the wavelengths of the SPEC4 channels."""
    def __init__(self):
        super().__init__()
        self.num_wavelengths = 6
    def get_wavelengths(self) -> Tensor:
        """Returns the wavelengths of the SPEC4 channels."""
        raise NotImplementedError
    def print_weights(self):
        for name, param in self.named_parameters():
            print(f'{name}: {param}')
    
class StandardWavelengthsModel(WavelengthsModel):
    """A model that predicts the standard wavelengths of the SPEC4 channels."""
    def __init__(self):
        super().__init__()
        self.wavelengths = nn.Parameter(conversions.spec4_wavelengths.clone())
    def get_wavelengths(self) -> Tensor:
        return self.wavelengths

class Model(nn.Module):
    """Base class for all models."""
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
    def print_weights(self):
        for name, param in self.named_parameters():
            print(f'{name}: {param}')
    
class LinearModel(Model):
    """A linear model."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
    def forward(self, x: Tensor) -> Tensor:
        y = self.linear(x)
        return y

class Fitting:
    """Base class for all fittings."""
    def __init__(self, name: str, in_channels: int, model_type: str, wavelengths_model_type: str, fit_wavelengths: bool):
        self.name = name
        self.in_channels = in_channels
        model_out_channels = 4
        if model_type == 'linear':
            self.model = LinearModel(in_channels, model_out_channels)
        else:
            raise ValueError(f'Unknown model type: {model_type}')
        if wavelengths_model_type == 'standard':
            self.wavelengths_model = StandardWavelengthsModel()
        else:
            raise ValueError(f'Unknown wavelengths model type: {wavelengths_model_type}')
        self.fit_wavelengths = fit_wavelengths
    def get_train_inputs(self, batch_size) -> Tensor:
        raise NotImplementedError
    def convert_from_xyz(self, xyz: Tensor) -> Tensor:
        raise NotImplementedError
    def get_reconstruction_loss(self, inputs: Tensor, outputs: Tensor) -> Tensor:
        """Penalizes not being able to reconstruct the input."""
        return nn.functional.mse_loss(inputs, outputs)
    def get_constraint_loss(self, spec4: Tensor) -> Tensor:
        """Penalizes not satisfying constraints: e.g., non-negativity."""
        positive_constraint_error = torch.relu(-spec4)
        squared_error = positive_constraint_error**2
        return squared_error.mean()
    def fit(self, num_steps: int = 40_000, batch_size: int = 1024, learning_rate: float = 0.0001, wavelengths_learning_rate: float = 0.01):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        if self.fit_wavelengths:
            self.wavelengths_model.train()
            optimizer.add_param_group({'params': self.wavelengths_model.parameters(), 'lr': wavelengths_learning_rate})
        else:
            self.wavelengths_model.eval()
        progress = tqdm(range(num_steps))
        progress.set_description(f'Fitting {self.name}')
        filtered_loss = 0
        for step in progress:
            inputs = self.get_train_inputs(batch_size)
            unclipped_spec4s = self.model(inputs)
            # closs = self.get_constraint_loss(unclipped_spec4s)
            clipped_spec4s = nn.functional.relu(unclipped_spec4s)
            padded_spec4s = torch.nn.functional.pad(clipped_spec4s, (1, 1), value=0.0)
            xyzs = conversions.batched_spectrum_to_XYZ(padded_spec4s, self.wavelengths_model.get_wavelengths())
            outputs = self.convert_from_xyz(xyzs)
            rloss = self.get_reconstruction_loss(inputs, outputs)
            loss = rloss #+ 0.0*closs
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
            filtered_loss = loss_val if step == 0 else 0.9 * filtered_loss + 0.1 * loss_val
            progress.set_postfix(loss=filtered_loss)
        print(f'Final loss: {filtered_loss:.16f}')

class RGBtoSPEC4Fitting(Fitting):
    def __init__(self):
        super().__init__('RGB to SPEC4', 3, model_type='linear', wavelengths_model_type='standard', fit_wavelengths=True)
    def get_train_inputs(self, batch_size) -> Tensor:
        srgb = torch.rand((batch_size, 3))
        rgb = conversions.batched_sRGB_to_RGB(srgb)
        return rgb
    def convert_from_xyz(self, xyz: Tensor) -> Tensor:
        return conversions.batched_XYZ_to_RGB(xyz)

if __name__ == '__main__':
    print("Fitting RGB to SPEC4...")
    rgb_to_spec4_fitting = RGBtoSPEC4Fitting()
    rgb_to_spec4_fitting.fit()
    print(f"{rgb_to_spec4_fitting.name} weights:")
    rgb_to_spec4_fitting.model.print_weights()
    print(f"{rgb_to_spec4_fitting.name} wavelengths:")
    print(repr(rgb_to_spec4_fitting.wavelengths_model.get_wavelengths()))
