"""This is the code used to determine the linear fits to convert to and from SPEC3"""
from typing import Optional
import torch
from torch import Tensor, nn
from tqdm.autonotebook import tqdm

import conversions

class WavelengthsModel(nn.Module):
    """A model that predicts the wavelengths of the SPEC3 channels."""
    def __init__(self):
        super().__init__()
        self.num_wavelengths = 6
    def get_wavelengths(self) -> Tensor:
        """Returns the wavelengths of the SPEC3 channels."""
        raise NotImplementedError
    def print_weights(self):
        for name, param in self.named_parameters():
            print(f'{name}: {param}')
    
class StandardWavelengthsModel(WavelengthsModel):
    """A model that predicts the standard wavelengths of the SPEC3 channels."""
    def __init__(self):
        super().__init__()
        self.wavelengths = nn.Parameter(conversions.SPEC3_standard_wavelengths.clone())
    def get_wavelengths(self) -> Tensor:
        return self.wavelengths

class PositiveWavelengthsModel(WavelengthsModel):
    """A model that predicts the standard wavelengths of the SPEC3 channels."""
    def __init__(self):
        super().__init__()
        wavelengths = conversions.SPEC3_standard_wavelengths
        log_dwavelengths = (wavelengths[1:] - wavelengths[:-1]).log()
        self.log_dwavelengths = nn.Parameter(log_dwavelengths.detach().clone())
        self.log_start_wavelength = nn.Parameter(wavelengths[0].log().detach().clone())
    def get_wavelengths(self) -> Tensor:
        dwavelengths = self.log_dwavelengths.exp()
        start_wavelength = self.log_start_wavelength.exp()
        wavelengths = torch.cat([
            start_wavelength.unsqueeze(0),
            start_wavelength + torch.cumsum(dwavelengths, dim=0),
        ], dim=0)
        return wavelengths
    
class Model(nn.Module):
    """Base class for all models."""
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
    def print_weights(self):
        torch.set_printoptions(precision=16)
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
    def postprocess_model_output(self, model_spec3: Tensor, extra: Optional[Tensor]) -> Tensor:
        return model_spec3
    def get_reconstruction_loss(self, inputs: Tensor, outputs: Tensor, extra: Optional[Tensor]) -> Tensor:
        """Penalizes not being able to reconstruct the input."""
        return nn.functional.mse_loss(inputs, outputs)
    def get_constraint_loss(self, spec3: Tensor, extra: Optional[Tensor]) -> Tensor:
        """Penalizes not satisfying constraints: e.g., non-negativity."""
        positive_constraint_error = torch.relu(-spec3)
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
            model_spec3s = self.model(inputs)
            unclipped_spec3s = self.postprocess_model_output(model_spec3s, extra)
            closs = self.get_constraint_loss(unclipped_spec3s, extra)
            padded_spec3s = torch.nn.functional.pad(unclipped_spec3s, (1, 1), value=0.0)
            xyzs = conversions.batched_spectrum_to_XYZ(padded_spec3s, self.wavelengths_model.get_wavelengths())
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
                model_spec3s = self.fitting.model(inputs)
                unclipped_spec3s = self.fitting.postprocess_model_output(model_spec3s, extra)
                clipped_spec3s = torch.nn.functional.relu(unclipped_spec3s)
                spec3s = clipped_spec3s
            inv_inputs = self.model(spec3s)
            rloss = self.get_reconstruction_loss(inputs, inv_inputs, extra)
            loss = rloss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
            filtered_loss = loss_val if step == 0 else 0.99 * filtered_loss + 0.01 * loss_val
            progress.set_postfix(loss=filtered_loss)
        print(f'Final loss: {filtered_loss:.16f}')

class RGBtoSPEC3Fitting(Fitting):
    def __init__(self):
        super().__init__('RGB to SPEC3', 3, model_type='linear', wavelengths_model_type='standard', fit_wavelengths=False)
    def get_train_inputs(self, batch_size) -> tuple[Tensor, Optional[Tensor]]:
        srgb = torch.rand((batch_size, 3))
        rgb = conversions.batched_sRGB_to_RGB(srgb)
        return rgb, None
    def convert_from_xyz(self, xyz: Tensor, extra: Optional[Tensor]) -> Tensor:
        return conversions.batched_XYZ_to_RGB(xyz)

class XYZtoSPEC3Fitting(Fitting):
    def __init__(self):
        super().__init__('XYZ to SPEC3', 3, model_type='linear', wavelengths_model_type='standard', fit_wavelengths=False)
    def get_train_inputs(self, batch_size) -> tuple[Tensor, Optional[Tensor]]:
        srgb = torch.rand((batch_size, 3))
        xyz = conversions.batched_sRGB_to_XYZ(srgb)
        return xyz, None
    def convert_from_xyz(self, xyz: Tensor, extra: Optional[Tensor]) -> Tensor:
        return xyz

class XYZStoSPEC3Fitting(Fitting):
    def __init__(self):
        super().__init__('XYZS to SPEC3', 3, model_type='linear', wavelengths_model_type='standard', fit_wavelengths=False)
    def get_train_inputs(self, batch_size) -> tuple[Tensor, Optional[Tensor]]:
        srgb = torch.rand((batch_size, 3))
        xyz = conversions.batched_sRGB_to_XYZ(srgb)
        s = torch.sum(xyz, dim=1, keepdim=True)
        xyz = xyz / s
        return xyz, s
    def postprocess_model_output(self, model_spec3: Tensor, extra: Optional[Tensor]) -> Tensor:
        return model_spec3 * extra
    def convert_from_xyz(self, xyz: Tensor, extra: Optional[Tensor]) -> Tensor:
        s = torch.sum(xyz, dim=1, keepdim=True)
        xyz = xyz / s
        return xyz

class XYZWavelengthFitting:
    """Finds optimal wavelengths that produce positive spectra for XYZ fitting."""
    def __init__(self, wavelengths_model_type: str = 'positive'):
        self.name = 'XYZ Wavelengths'
        self.model_type = wavelengths_model_type
        self.lambda_closs = 1.0
        if wavelengths_model_type == 'standard':
            self.wavelengths_model = StandardWavelengthsModel()
        elif wavelengths_model_type == 'positive':
            self.wavelengths_model = PositiveWavelengthsModel()
        else:
            raise ValueError(f'Unknown wavelengths model type: {wavelengths_model_type}')
    def get_train_inputs(self, batch_size) -> tuple[Tensor, Optional[Tensor]]:
        srgb = torch.rand((batch_size, 3))
        xyz = conversions.batched_sRGB_to_XYZ(srgb)
        return xyz, None
    def get_reconstruction_loss(self, inputs: Tensor, outputs: Tensor, extra: Optional[Tensor]) -> Tensor:
        """Penalizes not being able to reconstruct the input."""
        return nn.functional.mse_loss(inputs, outputs)
    def get_constraint_loss(self, spec3: Tensor, wavelengths: Tensor, extra: Optional[Tensor]) -> Tensor:
        """Penalizes not satisfying constraints: e.g., non-negativity."""
        positive_constraint_error = torch.relu(-spec3)
        spectrum_width = wavelengths[-1] - wavelengths[0]
        max_spectrum_width = 500.0
        max_spectrum_width_error = torch.relu(spectrum_width - max_spectrum_width)
        squared_error = positive_constraint_error**2 + max_spectrum_width_error**2
        return squared_error.mean()
    def get_XYZ_to_SPEC3_matrix(self, SPEC3_to_XYZ_matrix: Tensor) -> Tensor:
        return torch.inverse(SPEC3_to_XYZ_matrix)
    def fit(self, num_steps: int = 10_000, batch_size=8*1024, wavelengths_learning_rate: float = 0.001):
        self.wavelengths_model.train()
        optimizer = torch.optim.Adam(self.wavelengths_model.parameters(), lr=wavelengths_learning_rate)
        progress = tqdm(range(num_steps))
        progress.set_description(f'Fitting {self.name}')
        filtered_rloss = 0.0
        filtered_closs = 0.0
        for step in progress:
            with torch.no_grad():
                inputs, extra = self.get_train_inputs(batch_size)
            wavelengths = self.wavelengths_model.get_wavelengths()
            SPEC3_to_XYZ_matrix = conversions.get_optimal_SPEC3_to_XYZ_right_matrix(wavelengths)
            XYZ_to_SPEC3_matrix = self.get_XYZ_to_SPEC3_matrix(SPEC3_to_XYZ_matrix)
            unclipped_spec3s = torch.matmul(inputs, XYZ_to_SPEC3_matrix)
            clipped_spec3s = torch.nn.functional.relu(unclipped_spec3s)
            rec_xyzs = torch.matmul(clipped_spec3s, SPEC3_to_XYZ_matrix)
            rloss = self.get_reconstruction_loss(inputs, rec_xyzs, extra)
            closs = self.lambda_closs * self.get_constraint_loss(unclipped_spec3s, wavelengths, extra)
            loss = rloss + closs
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            rloss_val = rloss.item()
            closs_val = closs.item()
            filtered_closs = closs_val if step == 0 else 0.99 * filtered_closs + 0.01 * closs_val
            filtered_rloss = rloss_val if step == 0 else 0.99 * filtered_rloss + 0.01 * rloss_val
            progress.set_postfix(rloss=filtered_rloss, closs=filtered_closs)
        print(f'Final rloss: {filtered_rloss:.16f}')
        print(f'Final closs: {filtered_closs:.16f}')

if __name__ == '__main__':
    wfitting = XYZWavelengthFitting()
    print(f"Fitting {wfitting.name}...")
    wfitting.fit(num_steps=0)
    wfitting.wavelengths_model.eval()
    with torch.no_grad():
        torch.set_printoptions(precision=15, sci_mode=False)
        wavelengths = wfitting.wavelengths_model.get_wavelengths().detach().clone()
        SPEC3_to_XYZ_matrix = wfitting.get_SPEC3_to_XYZ_matrix(wavelengths)
        XYZ_to_SPEC3_matrix = wfitting.get_XYZ_to_SPEC3_matrix(SPEC3_to_XYZ_matrix)
        print()
        print(f"SPEC3_wavelengths = {repr(wavelengths)}")
        # print(f"XYZ_to_SPEC3_right_matrix = {repr(XYZ_to_SPEC3_matrix)}")
        print()
        print(f"SPEC3_to_XYZ_right_matrix = {repr(SPEC3_to_XYZ_matrix)}")
    
    fittings = [
        # RGBtoSPEC3Fitting(),
        # XYZtoSPEC3Fitting(),
    ]
    for fitting in fittings:
        print(f"Fitting {fitting.name}...")
        fitting.fit(num_steps=40_000)
        print(f"{fitting.name} weights:")
        fitting.model.print_weights()
        print(f"{fitting.name} wavelengths:")
        print(repr(fitting.wavelengths_model.get_wavelengths()))

        # inverse_fitting = InverseFitting(fitting)
        # print(f"Fitting {inverse_fitting.name}...")
        # inverse_fitting.fit(num_steps=100_000)
        # print(f"{inverse_fitting.name} weights:")
        # inverse_fitting.model.print_weights()
