# 3D Liver Shape Reconstruction Using Implicit Neural Representations
> This is an overview of the different implicit neural representations (INR) that aims to reconstruct 3D liver shapes. There are four models, including the baseline multilayer perceptron (MLP) model, fourier feature (FF) positional encoding model, neural radiance field (NeRF) positional encoding model, and sinusoidal representation (SIREN) model. The mesh was extracted using a Marching Cubes algorithm.

Laparoscopic liver surgery, commonly used to remove cancerous tumors, relies on minimal incisions to reduce postoperative complications but suffers from limited tactile feedback and visibility. As a result, precise pre-operative planning is crucial, often aided by MRI and CT scans. To enhance this process, our project focused on developing efficient deep learning models using implicit neural representations (INRs) to reconstruct 3D liver shapes. INRs model a continuous function that maps 3D coordinates to occupancy values, allowing for compact, accurate mesh generation from ground truth data. This method improves anatomical visualization and supports better surgical planning. Despite their potential, INRs still require optimization and thorough evaluation to ensure consistent and reliable performance. This project aims to create INRs that can depict the 3D liver shape as accurately as possible.

## Modules used

The following modules may need to be installed using the following command in Terminal:

```sh
! pip install <module>
```

* torch
* torch.nn
* torch.optim
* torch.nn.functional
* numpy
* torch.utils.tensorboard
* glob
* subprocess
* os
* trimesh
* tqdm 
* skimage.measure
* pathlib
* csv

## Dataset

There are a total of 452 3D liver meshes that were reconstructed.

## Models for one liver

This folder includes the python files that were used to generate a singular reconstructed 3D liver mesh. These files were used for initial testing and hyperparameter tuning phases. Please also set the current working directory so that it is compatible with with the code.

**ffpe.py** - This includes fourier feature positional encoding for the input layer.

**nerfpe.py** - This includes neural radiance field positional encoding for the input layer.

**siren.py** - This includes sinusoidal activation functions for each layer.

**standard_mlp.py** - This is the baseline multilayer perceptron model without any further enhancements or processing.

## Models for total liver

This folder includes the python files that were used to generate a comprehensive reconstruction of the entire 3D liver shape dataset. The INR training is iterated throughout the whole model and is stored into a separate folder at '/\<model\>_reconstructed_meshes'. The directory may need to be created using the following command in the case that the code does not run due to its absence:

```sh
cd <model>_reconstructed meshes
cd total_logs/occupancy_<model>
```

**ffpe_total.py** - This includes fourier feature positional encoding for the input layer.

**nerfpe_total.py** - This includes neural radiance field positional encoding for the input layer.

**siren_total.py** - This includes sinusoidal activation functions for each layer.

**standard_mlp_total.py** - This is the baseline multilayer perceptron model without any further enhancements or processing.

## Evaluation metrics

This folder includes all the python files required to analyze the 3D liver meshes that have been produced by the INR algorithm. It compares its respective ground truth mesh to the reconstructed mesh.

**mesh_utils.py** - This file contains all the evaluation metrics that were used, originally from Khoa Nguyen.

**single_liver_evaluation.py** - This file was used for hyperparameter tuning. A single mesh is assessed to produce a csv file containing all the evaluation metrics for each model.

**total_evaluation.py** - This file was used for evaluation across entire liver mesh dataset. Each iteration compares the ground truth mesh to its respective reconstructed mesh, then its evaluation metrics are recorded into the csv file. The average and standard deviations can be calculated after the csv file was created. It should be noted that the GPU environment is initiated at the start of each process. This can be adjusted according to the python environment the code is running in. Please also set the current working directory so that it is compatible with with the code.

## Result Files 

You can download all result files generated via this code [here](https://drive.google.com/file/d/143FM20plOQqEcY9EF0J9Wk2K2lGK-0hy/view?usp=sharing)
The folder contains the reconstructed liver meshes (.ply), the training loss curve saved from Tensorboard (.svg), and hyperparameter- tuning results (.csv). 
## Study Notes

This is a notebook file that includes the codes that were not used in the paper, but was used for analyzing the meshes during the development process. This file includes...

**Check memory used during inference** - This checks how much memory is being used while the process is running. This was monitored in case the memory usage exceed the maximum capacity.

**Post-processing: work after extracting mesh** - This includes lines of codes that can manipulate the mesh after it has been generated. This was not used in the final code.

**Trimesh object and sample points from the surface** - This includes the matplotlib module, where the 3D liver shape can be visualized in a graphical format.

**Check if vertices are inside the expected range** - This was used during debugging to check if the vertices generated by the Marching Cubes algorithm was within the range of -0.5 to 0.5.

## Further notes

The meshes are generated in a polygon file format (.ply), which can be visualized in the MeshLab application. 
