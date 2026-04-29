# Environment Setup Guide

This project requires Python 3.9+ and several scientific computing libraries for crystal structure analysis and generation.

## Setup Options

### Option 1: Using Conda (Recommended)

Create a new environment from the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate max-phase-paper
```

### Option 2: Using pip with pyproject.toml

```bash
pip install -e .
```

Or install dependencies only:

```bash
pip install -r requirements.txt
```

### Option 3: Using pip with requirements.txt

```bash
pip install -r requirements.txt
```

## Core Dependencies

| Package | Purpose | Version |
|---------|---------|---------|
| **pandas** | Data manipulation and analysis | ≥1.5.0 |
| **numpy** | Numerical computing | ≥1.23.0 |
| **scipy** | Scientific computing (statistics, etc.) | ≥1.9.0 |
| **pymatgen** | Crystallographic structure analysis | ≥2023.1.0 |
| **smact** | Chemistry validity screening | =3.0.1 |
| **matplotlib** | Static visualizations | ≥3.5.0 |
| **bokeh** | Interactive visualizations | ≥3.0.0 |
| **jupyter** | Notebook support | ≥1.0.0 |
| **tqdm** | Progress bars | ≥4.60.0 |
|**geckodriver** | Renders heatmap | 0.36.0 |

## Verify Installation

Test the installation by starting a Jupyter notebook:

```bash
jupyter notebook MAX_phases_data_reproduction.ipynb
```

Or verify imports in Python:

```python
import pandas as pd
import numpy as np
import scipy
import pymatgen
import smact
import matplotlib.pyplot as plt
import bokeh
from tqdm import tqdm

print("All dependencies installed successfully!")
```

## Module Structure

The project includes the following modules in `src/`:

- **max_validation.py** — MAX phase formula validation
- **curvature.py** — A-site well curvature calculations
- **stability_sweep.py** — Stability sweep analysis with Fisher's exact test
- **periodic_heatmap.py** — Periodic table heatmap visualizations
- **generate_prompts.py** — MAX/MAB phase prompt generation
- **data_key.py** — Dataset reference key (to be configured)