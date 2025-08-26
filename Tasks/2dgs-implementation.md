# 2DGS Integration into Nerfstudio

This document outlines the steps taken to integrate the 2D Gaussian Splatting (2DGS) model into the Nerfstudio framework.

## 1. Create the New Model File

- A new file, `nerfstudio/models/twodgsfacto.py`, was created to house the implementation of the 2DGS model.

## 2. Define the Model Configuration

- A `TwoDGSfactoModelConfig` dataclass was defined in `nerfstudio/models/twodgsfacto.py`, inheriting from `SplatfactoModelConfig`.
- A new parameter, `normal_loss_lambda`, was added to the configuration to control the weight of the normal consistency loss.

## 3. Implement the Model Class

- A `TwoDGSfactoModel` class was implemented in `nerfstudio/models/twodgsfacto.py`, inheriting from `SplatfactoModel`.

## 4. Override the Rendering Method

- The `get_outputs` method was overridden to replace the call to `gsplat.rendering.rasterization` with `gsplat.rendering.rasterization_2dgs`.

## 5. Implement the Custom Loss

- The `get_loss_dict` method was overridden to include the `normal_loss` from the 2DGS implementation.

## 6. Create a New Configuration File

- A new configuration for `twodgsfacto` was added to `nerfstudio/configs/method_configs.py` to make the model accessible via the `ns-train` command.

## 7. Register the New Model

- The `TwoDGSfactoModelConfig` was imported in `nerfstudio/configs/method_configs.py` to register the new model.
