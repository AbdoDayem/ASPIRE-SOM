# Density Tracking Self-Organizing Map (DTSOM)

This script contains a variant of the regular SOM, which implements a naive attempt at Density Tracking. The goal is to create a topology that tracks the denseness of regions and influences data points to cluster at certain sectors. The script involves implementation of the default way and accelerated through the GPU. Moreover, the script contains command-line arguments for the iterations to use, position to save the maps, and which model to train.

## Features

- **DTSOM_CPU**: The default variant, based on CPU.
- **DTSOM_GPU**: An accelerated version of DTSOM that uses the GPU.
- **Command-Line Arguments**:
  - **`--acc`**: Use GPU (DTSOM_GPU) for training.
  - **`--it [number]`**: Set the number of iterations for training (default: 10).
  - **`--name [name]`**: Specify a custom name to save the SOMs (default: `dt_test2`).

## Prerequisites

Before running the script, make sure you have the necessary dependencies installed. You can install them via `pip`:

```powershell
pip install -r requirements.txt
```

You will need:

- Python 3.x
- Dependencies listed in `requirements.txt`

If you want to use **DTSOM_GPU**, ensure to have CUDA and the `cupy` library installed for GPU support.

## Usage

You can run the script from the command line and provide different options via command-line arguments.

### Arguments

- **`--acc`**: Use GPU (DTSOM_GPU) for training.
- **`--it [number]`**: Set the number of iterations for training (default: 10).
- **`--name [name]`**: Specify a custom name to save the SOMs (default: `dt_test2`).

### Example 1: Use GPU Acceleration, Set Iterations to 50, and Specify a Custom SOM Name

```powershell
python script.py --acc --it 50 --name acc_som
```

- This will use `DTSOM_GPU` for training, set the number of iterations to 50, and save the trained models with the name `acc_som`.

### Example 2: Use Default Settings (CPU, 10 Iterations, Default SOM Name)

```powershell
python script.py
```

- This will use `DTSOM_CPU` by default, set iterations to 10, and save the SOM models with the name `dt_test2`.

### Example 3: Use CPU, Set Iterations to 20, and Specify a Custom SOM Name

```bash
python script.py --it 20 --name cpu_som
```

- This will use `DTSOM_CPU`, set iterations to 20, and save the trained models with the name `cpu_som`.
