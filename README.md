# 3D Forward Simulation of Gravity Anomaly Data

This repository contains a complete implementation of the SimPEG tutorial for simulating 3D gravity anomaly data using the integral formulation on a tensor mesh.

## Overview

This tutorial demonstrates:
- ✅ How to simulate gravity data for 3D structures with SimPEG
- ✅ How to create gravity surveys
- ✅ How to design a tensor mesh for gravity simulation
- ✅ How to predict gravity anomaly data for a density contrast model
- ✅ How to include surface topography in the forward simulation
- ✅ Understanding units of density contrast models and resulting data

## Features

The implementation includes:
- **Topography Generation**: Synthetic surface topography with random variations
- **Gravity Survey Setup**: 289 observation points located 5m above topography
- **Tensor Mesh Design**: Optimized mesh with padding for accurate simulations
- **Density Model**: Less dense block and more dense sphere structures
- **Forward Simulation**: Fast Choclo engine for efficient computation
- **Visualization**: 3D topography, density model cross-sections, and gravity anomaly maps
- **Data Export**: Optional export of topography and gravity data with noise

## Installation

### Requirements

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Package Dependencies

- `simpeg>=0.21.0` - Simulation and Parameter Estimation in Geophysics
- `discretize>=0.10.0` - Mesh generation and discretization
- `choclo>=0.1.0` - Fast gravity/magnetic forward modeling
- `numpy>=1.20.0` - Numerical computing
- `scipy>=1.7.0` - Scientific computing
- `matplotlib>=3.5.0` - Plotting and visualization
- `numba>=0.55.0` - JIT compilation for performance

## Usage

### Basic Usage

Run the tutorial script:

```bash
python gravity_anomaly_3d_tutorial.py
```

This will:
1. Generate synthetic topography
2. Create a gravity survey with 289 observation points
3. Build a tensor mesh with ~50,000 cells
4. Define a density contrast model (block + sphere)
5. Run the forward simulation using Choclo engine
6. Generate three visualization plots

### Output Files

The script generates the following PNG files:

1. **topography.png** - 3D visualization of surface topography
2. **density_model.png** - Y=0 cross-section showing density contrasts
3. **gravity_anomaly.png** - Map view of simulated gravity anomaly data

### Data Export (Optional)

To export topography and gravity data files, modify the script:

```python
save_output = True  # Change from False to True
```

This creates a `fwd_gravity_anomaly_3d_outputs/` directory with:
- `gravity_topo.txt` - Topography XYZ coordinates
- `gravity_data.obs` - Gravity data with 2% noise (X, Y, Z, gz)

## Understanding the Results

### Coordinate System

SimPEG uses a **right-handed coordinate system with Z positive upward**:
- **Negative anomalies** (blue): Stronger gravity → More dense material below
- **Positive anomalies** (red): Weaker gravity → Less dense material below

### Model Parameters

- **Mesh**: 50,000 cells with 5m core cell size
- **Active cells**: ~25,000 (cells below topography)
- **Background density**: 0.0 g/cc
- **Block density**: -0.2 g/cc (less dense)
- **Sphere density**: +0.2 g/cc (more dense)

### Simulation Details

- **Engine**: Choclo (Numba-accelerated)
- **Method**: 3D integral formulation
- **Component**: gz (vertical gravity anomaly)
- **Units**: mGal (milligals)
- **Sensitivity storage**: forward_only (memory efficient)

## Code Structure

The script is organized into clear sections:

```
1. Import Modules
2. Define Topography
3. Define Survey (receivers, sources)
4. Design Tensor Mesh
5. Define Active Cells
6. Create Model Mapping
7. Define Density Model
8. Setup Forward Simulation
9. Simulate Gravity Data
10. Export Data (optional)
```

## Troubleshooting

### Installation Issues

If you encounter issues installing `choclo`:
```bash
pip install numba  # Install numba first
pip install choclo
```

### Memory Issues

For large meshes, reduce mesh size by modifying:
```python
dh = 10.0  # Increase from 5.0 to 10.0
```

### Visualization Issues

If plots don't appear, ensure you have a display backend:
```bash
# On Linux without display
export MPLBACKEND=Agg
```

## References

- [SimPEG Documentation](https://docs.simpeg.xyz/)
- [Original Tutorial](https://docs.simpeg.xyz/latest/content/tutorials/03-gravity/plot_1a_gravity_anomaly.html)
- [Choclo Library](https://github.com/fatiando/choclo)
- [Discretize Package](https://discretize.simpeg.xyz/)

## Learning Objectives

After running this tutorial, you will understand:

1. ✅ How to create gravity surveys with receivers and sources
2. ✅ How to design appropriate meshes for gravity forward modeling
3. ✅ How to define density contrast models on irregular topography
4. ✅ How to run efficient forward simulations with the Choclo engine
5. ✅ How to interpret gravity anomaly data (sign conventions)
6. ✅ How to visualize 3D geophysical models and data

## Keywords

gravity survey, gravity anomaly, forward simulation, integral formulation, tensor mesh, SimPEG, Choclo, geophysics, potential fields

## License

This tutorial is based on the SimPEG documentation and examples, which are available under the MIT license.

## Contact

For issues or questions about SimPEG:
- [SimPEG GitHub](https://github.com/simpeg/simpeg)
- [SimPEG Discourse](https://simpeg.discourse.group/)

For issues with this implementation, please check the tutorial against the latest SimPEG documentation.
