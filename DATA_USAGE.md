# Crystal Structure Dataset Usage Guide

The `data_key.py` module provides a smart, selective data loading system that works with both local files and Hugging Face Hub.

## Quick Start

### List All Available Datasets

```python
from src.data_key import list_datasets, get_loader

# See what's available
datasets = list_datasets()
for name, description in datasets.items():
    print(f"{name:25s} → {description}")
```

### Load Specific Data

**Local mode (development):**
```python
from src.data_key import get_loader

loader = get_loader(mode='local') # if already downloaded

# Load just what you need
df = loader.load_parquets('MACE_screening')
print(f"Loaded {len(df):,} structures")
```

**Hugging Face mode (download only what you need):**
```python
loader = get_loader(mode='hf')

# Auto-downloads only the MACE_screening data from HF
df = loader.load_parquets('MACE_screening')
print(f"Loaded {len(df):,} structures")
```

**Auto-detect mode:**
```python
loader = get_loader()  # Uses hf if available, else local

df = loader.load_parquets('PKV_MACE')
```

## Workflow

### 1. Explore Available Data Without Downloading

```python
loader = get_loader(mode='hf')

# Get metadata only (no download!)
metadata = loader.get_metadata('non_conditional')
print(metadata)
# {
#   'description': 'Baseline non-conditional generation results',
#   'total_structures': 193_001,
#   'subdirectories': {...},
#   ...
# }
```

### 2. Load Configuration-Specific Data

```python
loader = get_loader()

# Access nested configuration data
pkv_config = loader.get_config('perturbation_experiments', 'PKV')
print(f"PKV structures: {pkv_config['structures_generated']}")

# Load the sweep data
sweep_df = loader.load_parquets('perturbation_experiments')  # Will include PKV
```

### 3. Work with Multiple Datasets

```python
loader = get_loader()

# Load different experiment types
baseline_df = loader.load_parquets('base_corpus')
mace_pkv = loader.load_parquets('MACE_screening')  # PKV results
mace_slider = loader.load_parquets('MACE_screening')  # Slider results

# Combine as needed
combined = pd.concat([baseline_df, mace_pkv], ignore_index=True)
```

### 4. Selective Download (HF Mode Only)

The beauty of this approach: when using HF mode, only the datasets you actually load are downloaded. Everything is cached locally, so repeated access is instant.

```python
loader = get_loader(mode='hf')

df1 = loader.load_parquets('non_conditional')

df2 = loader.load_parquets('non_conditional')

df3 = loader.load_parquets('PKV_CIFs')

# Doesn't download other datasets unless specified
```

## Dataset Structure

### Top-Level Datasets (Single Dataset)

Load directly with `load_parquets()`:

- `base_corpus` — LeMaterial cleaned training data (4.3M structures)
- `first_matches` — Small curated validation set (509 structures)
- `boride_candidates` — Initial boride candidate pool
- `non_conditional` — Non-conditional generation results (193K structures)
- `conditional_generation` — Conditional results (65K structures)
- `updated_results` — Latest refined results (369K structures)
- `updated_MACE` — Latest MACE screening (8.9M structures - note this includes duplications as mentioned previously.)

### Nested Datasets (Get Config First)

Explore with `get_config()`:

- `perturbation_experiments` — A-site sweep experiments (PKV, slider, nc)
- `CIF_generation` — CIF outputs by method (PKV, slider)
- `MACE_screening` — ML screening results (PKV, slider)
- `non_comp_analysis` — Non-compositional analysis 

```python
# Get configuration details
config = loader.get_config('perturbation_experiments', 'PKV')
print(f"Structures generated: {config['structures_generated']}")

# Access the sweep data
sweep_path = config['A_site_sweep']  # Points to specific parquet
```

### Custom Local Path

```python
loader = get_loader(mode='local', local_path='/path/to/custom/data')
df = loader.load_parquets('PKV_MACE')
```

### Specify HF Cache Location

```python
loader = get_loader(
    mode='hf',
    cache_dir='/custom/cache/location'
)
```
## Troubleshooting

### "Dataset not found" Error

Make sure the dataset name is valid:

```python
loader = get_loader()
available = loader.list_available()
print(f"Valid datasets: {list(available.keys())}")
```

### "No parquet files found" Error

The dataset exists but has no parquet files in that specific path. Check the metadata:

```python
meta = loader.get_metadata('dataset_name')
print(f"HF path: {meta.get('hf_path')}")
print(f"Subdirectories: {meta.get('subdirectories')}")
```

### Slow First Load in HF Mode

First load downloads from HF (~50 MB/s depending on network). Subsequent loads are instant (cached).

### Cache Getting Too Large

Clear the HF cache:

```bash
rm -rf ~/.cache/huggingface/datasets
```

Or use custom cache location:

```python
loader = get_loader(cache_dir='/tmp/hf_cache')
```

## Integration in Notebooks

```python
# At the top of notebook
from src.data_key import get_loader
import pandas as pd

# Initialise once
loader = get_loader()

# Use throughout analysis
def load_experiment_data(exp_name):
    """Helper to load experiment-specific data."""
    return loader.load_parquets(exp_name)

# Now you can do:
df_pkv = load_experiment_data('PKV_MACE')
df_baseline = load_experiment_data('non_conditional')
```
