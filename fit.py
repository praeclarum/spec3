"""This is the code used to determine the linear fits to convert to and from SPEC4"""
from typing import Optional
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
        self.model_type = model_type
        if model_type == 'linear':
            self.model = LinearModel(in_channels, model_out_channels)
        else:
            raise ValueError(f'Unknown model type: {model_type}')
        if wavelengths_model_type == 'standard':
            self.wavelengths_model = StandardWavelengthsModel()
        else:
            raise ValueError(f'Unknown wavelengths model type: {wavelengths_model_type}')
        self.fit_wavelengths = fit_wavelengths
    def get_train_inputs(self, batch_size) -> tuple[Tensor, Optional[Tensor]]:
        raise NotImplementedError
    def convert_from_xyz(self, xyz: Tensor, extra: Optional[Tensor]) -> Tensor:
        raise NotImplementedError
    def postprocess_model_output(self, model_spec4: Tensor, extra: Optional[Tensor]) -> Tensor:
        return model_spec4
    def get_reconstruction_loss(self, inputs: Tensor, outputs: Tensor, extra: Optional[Tensor]) -> Tensor:
        """Penalizes not being able to reconstruct the input."""
        return nn.functional.mse_loss(inputs, outputs)
    def get_constraint_loss(self, spec4: Tensor, extra: Optional[Tensor]) -> Tensor:
        """Penalizes not satisfying constraints: e.g., non-negativity."""
        positive_constraint_error = torch.relu(-spec4)
        squared_error = positive_constraint_error**2
        return squared_error.mean()
    def fit(self, num_steps: int = 10_000, batch_size: int = 4*1024, learning_rate: float = 0.0001, wavelengths_learning_rate: float = 0.01):
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
            inputs, extra = self.get_train_inputs(batch_size)
            model_spec4s = self.model(inputs)
            unclipped_spec4s = self.postprocess_model_output(model_spec4s, extra)
            closs = self.get_constraint_loss(unclipped_spec4s, extra)
            padded_spec4s = torch.nn.functional.pad(unclipped_spec4s, (1, 1), value=0.0)
            xyzs = conversions.batched_spectrum_to_XYZ(padded_spec4s, self.wavelengths_model.get_wavelengths())
            outputs = self.convert_from_xyz(xyzs, extra)
            rloss = self.get_reconstruction_loss(inputs, outputs, extra)
            loss = rloss + 1000.0*closs
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
            filtered_loss = loss_val if step == 0 else 0.9 * filtered_loss + 0.1 * loss_val
            progress.set_postfix(loss=filtered_loss)
        print(f'Final loss: {filtered_loss:.16f}')

class InverseFitting:
    """Base class for all inverse fittings."""
    def __init__(self, fitting: Fitting):
        self.name = f"Inverse {fitting.name}"
        self.fitting = fitting
        self.in_channels = 4
        model_out_channels = fitting.in_channels
        if fitting.model_type == 'linear':
            self.model = LinearModel(self.in_channels, model_out_channels)
        else:
            raise ValueError(f'Unknown model type: {fitting.model_type}')
        self.wavelengths_model = fitting.wavelengths_model
    def get_reconstruction_loss(self, inputs: Tensor, outputs: Tensor, extra: Optional[Tensor]) -> Tensor:
        """Penalizes not being able to reconstruct the input."""
        return nn.functional.mse_loss(inputs, outputs)
    def fit(self, num_steps: int = 40_000, batch_size: int = 1024, learning_rate: float = 0.0001, wavelengths_learning_rate: float = 0.01):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.wavelengths_model.eval()
        self.fitting.model.eval()
        progress = tqdm(range(num_steps))
        progress.set_description(f'Fitting {self.name}')
        filtered_loss = 0
        for step in progress:
            with torch.no_grad():
                inputs, extra = self.fitting.get_train_inputs(batch_size)
                model_spec4s = self.fitting.model(inputs)
                unclipped_spec4s = self.fitting.postprocess_model_output(model_spec4s, extra)
                clipped_spec4s = torch.nn.functional.relu(unclipped_spec4s)
                spec4s = clipped_spec4s
            inv_inputs = self.model(spec4s)
            rloss = self.get_reconstruction_loss(inputs, inv_inputs, extra)
            loss = rloss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
            filtered_loss = loss_val if step == 0 else 0.99 * filtered_loss + 0.01 * loss_val
            progress.set_postfix(loss=filtered_loss)
        print(f'Final loss: {filtered_loss:.16f}')

class RGBtoSPEC4Fitting(Fitting):
    def __init__(self):
        super().__init__('RGB to SPEC4', 3, model_type='linear', wavelengths_model_type='standard', fit_wavelengths=False)
    def get_train_inputs(self, batch_size) -> tuple[Tensor, Optional[Tensor]]:
        srgb = torch.rand((batch_size, 3))
        rgb = conversions.batched_sRGB_to_RGB(srgb)
        return rgb, None
    def convert_from_xyz(self, xyz: Tensor, extra: Optional[Tensor]) -> Tensor:
        return conversions.batched_XYZ_to_RGB(xyz)

class XYZStoSPEC4Fitting(Fitting):
    def __init__(self):
        super().__init__('XYZS to SPEC4', 3, model_type='linear', wavelengths_model_type='standard', fit_wavelengths=False)
    def get_train_inputs(self, batch_size) -> tuple[Tensor, Optional[Tensor]]:
        srgb = torch.rand((batch_size, 3))
        xyz = conversions.batched_sRGB_to_XYZ(srgb)
        s = torch.sum(xyz, dim=1, keepdim=True)
        xyz = xyz / s
        return xyz, s
    def postprocess_model_output(self, model_spec4: Tensor, extra: Optional[Tensor]) -> Tensor:
        return model_spec4 * extra
    def convert_from_xyz(self, xyz: Tensor, extra: Optional[Tensor]) -> Tensor:
        s = torch.sum(xyz, dim=1, keepdim=True)
        xyz = xyz / s
        return xyz

if __name__ == '__main__':
    fitting = RGBtoSPEC4Fitting()
    # fitting = XYZStoSPEC4Fitting()
    print(f"Fitting {fitting.name}...")
    fitting.fit(num_steps=20_000)
    print(f"{fitting.name} weights:")
    fitting.model.print_weights()
    print(f"{fitting.name} wavelengths:")
    print(repr(fitting.wavelengths_model.get_wavelengths()))

    inverse_fitting = InverseFitting(fitting)
    print(f"Fitting {inverse_fitting.name}...")
    inverse_fitting.fit(num_steps=20_000)
    print(f"{inverse_fitting.name} weights:")
    inverse_fitting.model.print_weights()
