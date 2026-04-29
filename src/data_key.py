"""
Crystal structure dataset reference and loader.

Supports both local filesystem and Hugging Face Hub access. Use the DatasetLoader
class for selective downloading and intuitive data access.

Example:
    Local mode (development):
        loader = DatasetLoader(mode='local', local_path='/path/to/data')
        df = loader.load_parquets('PKV_MACE')

    Hugging Face mode (shared):
        loader = DatasetLoader(mode='hf')
        df = loader.load_parquets('PKV_MACE')  # Auto-downloads to cache
        metadata = loader.get_metadata('PKV_MACE')
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import pandas as pd
    from huggingface_hub import snapshot_download
except ImportError:
    pd = None
    snapshot_download = None

# ============================================================================
# DATASET METADATA & STRUCTURE
# ============================================================================

DATASET_KEY = {
    "base_corpus": {
        "description": "LeMaterial cleaned baseline dataset used as the base training corpus",
        "hf_path": "lematerial_clean/data",
        "local_subpath": "lematerial_clean/data",
        "total_structures": 4_289_372,
        "use_case": "Training data for baseline models",
    },

    "first_match_subset": {
        "description": "A small first match analysis",
        "hf_path": "ALL_DATA/first_matches",
        "local_subpath": "first_matches",
        "total_structures": 509,
        "use_case": "Manual validation and review",
    },

    "boride_candidates": {
        "description": "Boride candidates selected using MACE to be passed to DFT for further investigation.",
        "hf_path": "boride_candidates.parquet",
        "local_subpath": "boride_candidates.parquet",
        "file_type": "parquet",
    },

    "dataset_with_curvature": {
        "description": "Fine-tuning dataset with imputed A-site curvature values",
        "hf_path": "dataset_with_curvature_imputed.parquet",
        "local_subpath": "dataset_with_curvature_imputed.parquet",
        "file_type": "parquet",
    },

    "novel_stable_structures": {
        "PKV": {
            "description": "Novel stable structures from PKV conditional generation",
            "hf_path": "ALL_DATA/pkv_novel_stable_configs.parquet",
            "local_subpath": "pkv_novel_stable_configs.parquet",
            "file_type": "parquet",
        },
        "slider": {
            "description": "Novel stable structures from slider conditional generation",
            "hf_path": "ALL_DATA/slider_novel_stable_configs.parquet",
            "local_subpath": "slider_novel_stable_configs.parquet",
            "file_type": "parquet",
        },
        "baseline": {
            "description": "Novel stable structures from baseline non-conditional generation",
            "hf_path": "baseline_novel_stable_configs.parquet",
            "local_subpath": "baseline_novel_stable_configs.parquet",
            "file_type": "parquet",
        },
    },

    "MACE_candidates": {
        "description": "MACE-screened compositionally novel candidates for DFT validation",
        "hf_path": "MACE_screened_candidates.parquet",
        "local_subpath": "MACE_screened_candidates.parquet",
        "file_type": "parquet",
    },

    "DFT_validation": {
        "MAX": {
            "description": "DFT-validated MAX phase candidates",
            "hf_path": "MAX_dft_proc.parquet",
            "local_subpath": "MAX_dft_proc.parquet",
            "file_type": "parquet",
        },
        "MAB": {
            "description": "DFT-validated MAB phase candidates",
            "hf_path": "MAB_dft_proc.parquet",
            "local_subpath": "MAB_dft_proc.parquet",
            "file_type": "parquet",
        },
    },

    "perturbation_experiments": {
        "PKV": {
            "description": "A-site perturbation to establish the curvature condition vector",
            "A_site_sweep": "ALL_DATA/PKV_perturb_A_site_sweep.parquet",
            "A_site_results": "PKV_A_site/A_site",
            "structures_generated": 22_989,
        },
        "slider": {
            "description": "A-site perturbation to establish the curvature condition vector",
            "A_site_sweep": "ALL_DATA/slider_perturb_A_site_sweep.parquet",
            "A_site_results": "slider_A_site/A_site",
            "structures_generated": 21_999,
        },
        "nc": {
            "description": "A-site perturbation to establish the curvature condition vector for non-conditional generation",
            "A_site_sweep": "ALL_DATA/nc_perturb_A_site_sweep.parquet",
            "A_site_results": "nc_A_site/A_site",
            "structures_generated": 24_387,
        },
    },

    "CIF_generation": {
        "PKV": {
            "description": "CIF files generated from PKV experiments. Tagged with the condition vector for each file in the form `*0p00_0p00*` corresponding to normalised MXene derivative count and A-site curvature respectively.",
            "hf_path": "ALL_DATA/PKV_CIFs",
            "local_subpath": "PKV_CIFs",
            "total_structures": 124_724,
        },
        "slider": {
            "description": "CIF files generated from slider experiments. Tagged with the condition vector for each file in the form `*0p00_0p00*` corresponding to normalised MXene derivative count and A-site curvature respectively.",
            "hf_path": "ALL_DATA/slider_CIFs",
            "local_subpath": "slider_CIFs",
            "total_structures": 120_247,
        },
    },

    "MACE_screening": {
        "PKV": {
            "description": "MACE ML screening results for PKV",
            "hf_path": "ALL_DATA/PKV_MACE",
            "local_subpath": "PKV_MACE",
            "total_structures": 9_314,
        },
        "slider": {
            "description": "MACE ML screening results for slider",
            "hf_path": "ALL_DATA/slider_MACE",
            "local_subpath": "slider_MACE",
            "total_structures": 8_420,
        },
    },

    "non_conditional": {
        "description": "Baseline non-conditional generation results",
        "hf_path": "ALL_DATA/non-conditional",
        "local_subpath": "ALL_DATA/non-conditional",
        "total_structures": 193_001,
        "subdirectories": {
            "MACE": 21_020,
            "cifs": 50_939,
            "metrics": 24_385,
            "postprocessed": 20_720,
            "prompts": 3_552,
        },
    },

    "conditional_generation": {
        "description": "Conditional generation results",
        "hf_path": "ALL_DATA/updated_results/conditional",
        "local_subpath": "ALL_DATA/updated_results/conditional",
        "total_structures": 65_464,
        "subdirectories": {
            "postprocessed": 50_649,
            "prompts": 14_815,
        },
    },

    "non_comp_analysis": {
        "description": "Complementarity/competition analysis across configurations",
        "hf_path": "ALL_DATA/non_comp",
        "local_subpath": "ALL_DATA/non_comp",
        "total_structures": 5_994_174,
        "configurations": {
            "conditional": 4_158,
            "non-conditional": 5_859_098,
            "slider": 5_114,
        },
    },

    "updated_results": {
        "description": "Latest refined results across all configurations",
        "hf_path": "ALL_DATA/updated_results",
        "local_subpath": "ALL_DATA/updated_results",
        "total_structures": 369_530,
        "configurations": {
            "PKV": {"structures": 22_977},
            "nc": {"structures": 1_341},
            "slider": {"structures": 22_051},
        },
    },

    "updated_MACE": {
        "description": "Latest MACE screening results",
        "hf_path": "ALL_DATA/updated_MACE",
        "local_subpath": "ALL_DATA/updated_MACE",
        "total_structures": 8_951_368,
    },

    "summary_statistics": {
        "total_generations": 17_230_356,
        "total_files": 816,
        "total_size_gb": 1.5,
        "project_name": "MAX Phase Crystal Structure Generation",
        "project_description": "Comprehensive crystal structure generation for MAX phases",
    },
}

# ============================================================================
# DATASET LOADER
# ============================================================================


class DatasetLoader:
    """Load datasets from Hugging Face Hub (primary) or local filesystem with selective downloading."""

    def __init__(
        self,
        mode: str = "hf",
        local_path: Optional[str] = None,
        hf_repo: str = "Jamie1701/conditional-generative-models-max-phase",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize dataset loader.

        Parameters:
        -----------
        mode : str
            'local' (use local filesystem), 'hf' (use Hugging Face - default), or 'auto' (try local first, fallback to HF)
        local_path : str, optional
            Path to local dataset root. Only used if mode='local'
        hf_repo : str
            Hugging Face repo ID
        cache_dir : str, optional
            Where to cache HF downloads. Defaults to ~/.cache/huggingface
        """
        self.hf_repo = hf_repo
        self.cache_dir = cache_dir or os.path.expanduser(
            "~/.cache/huggingface/datasets")

        # Handle mode selection
        if mode == "auto":
            # Try local first, but fall back to HF if not available
            if local_path and Path(local_path).exists():
                mode = "local"
            elif self._find_local_dataset():
                mode = "local"
                local_path = str(self._find_local_dataset())
            else:
                mode = "hf"

        self.mode = mode

        # Set local_path only if we're using local mode
        if self.mode == "local":
            if not local_path:
                local_path = str(self._find_local_dataset())
            if not Path(local_path).exists():
                raise ValueError(
                    f"Local mode selected but dataset not found at {local_path}")
            self.local_path = Path(local_path)
        else:
            self.local_path = None

    @staticmethod
    def _find_local_dataset() -> Optional[Path]:
        """Find local ALL_DATA directory."""
        candidates = [
            Path.home() / "Documents/AWE/ALL_DATA",
            Path("/Users/jamiepersonal/Documents/AWE/ALL_DATA"),
            Path.cwd() / "ALL_DATA",
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def get_metadata(self, dataset_name: str) -> Dict[str, Any]:
        """Get metadata for a dataset without downloading."""
        if dataset_name not in DATASET_KEY:
            raise KeyError(
                f"Dataset '{dataset_name}' not found. Available: {list(DATASET_KEY.keys())}")
        return DATASET_KEY[dataset_name]

    def list_available(self) -> Dict[str, Dict[str, Any]]:
        """List all available datasets with descriptions."""
        available = {}
        for name, meta in DATASET_KEY.items():
            if isinstance(meta, dict) and "description" in meta:
                available[name] = {
                    "description": meta.get("description"),
                    "total_structures": meta.get("total_structures"),
                    "total_size_gb": meta.get("total_size_gb"),
                }
        return available

    def _get_path(self, dataset_name: str) -> Path:
        """Get local path to dataset, downloading from HF if needed."""
        meta = self.get_metadata(dataset_name)

        if self.mode == "local":
            subpath = meta.get("local_subpath") or meta.get("hf_path")
            return self.local_path / subpath

        else:  # mode == 'hf'
            hf_path = meta.get("hf_path") or meta.get("local_subpath")
            if not hf_path:
                raise ValueError(f"No HF path for dataset '{dataset_name}'")

            # Download only this specific dataset
            cache_root = Path(self.cache_dir) / self.hf_repo.replace("/", "_")
            local_dir = cache_root / \
                hf_path.split("/")[0] if "/" in hf_path else cache_root

            if snapshot_download is None:
                raise ImportError(
                    "huggingface_hub not installed. Install with: pip install huggingface_hub")

            try:
                snapshot_download(
                    self.hf_repo,
                    repo_type="dataset",
                    local_dir=cache_root,
                    allow_patterns=f"{hf_path}/**",
                    cache_dir=self.cache_dir,
                )
            except Exception as e:
                print(f"Warning: Could not download {dataset_name}: {e}")

            return cache_root / hf_path

    def load_parquets(self, dataset_name: str):
        """Load all parquet files from a dataset into a DataFrame."""
        if pd is None:
            raise ImportError(
                "pandas not installed. Install with: pip install pandas")

        path = self._get_path(dataset_name)

        if not path.exists():
            raise FileNotFoundError(f"Dataset path not found: {path}")

        # Handle single parquet file
        if path.suffix == ".parquet":
            return pd.read_parquet(path)

        # Handle directory of parquet files
        parquet_files = list(path.glob("*.parquet"))

        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {path}")

        dfs = [pd.read_parquet(f) for f in sorted(parquet_files)]
        return pd.concat(dfs, ignore_index=True)

    def get_config(self, dataset_name: str, config_name: str) -> Dict[str, Any]:
        """Get configuration details for nested datasets (e.g., 'perturbation_experiments.PKV')."""
        meta = DATASET_KEY.get(dataset_name)
        if not meta or config_name not in meta:
            raise KeyError(
                f"Config '{config_name}' not found in '{dataset_name}'")
        return meta[config_name]


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_loader(mode: str = "hf") -> DatasetLoader:
    """
    Get a DatasetLoader instance with default settings.

    Args:
        mode: 'hf' (default - Hugging Face), 'local' (local filesystem), or 'auto' (try local first)
    """
    return DatasetLoader(mode=mode)


def load_dataset(dataset_name: str, config_name: Optional[str] = None):
    """
    Load a dataset using DatasetLoader with direct path handling.
    Handles simple datasets and nested configurations.

    Args:
        dataset_name: Name of the dataset (e.g., 'MACE_candidates', 'perturbation_experiments')
        config_name: Optional configuration for nested datasets (e.g., 'PKV')

    Returns:
        pandas DataFrame with the loaded data
    """
    if pd is None:
        raise ImportError(
            "pandas not installed. Install with: pip install pandas")

    loader = get_loader()

    if config_name:
        # For nested datasets, get the specific file path from the config
        config = loader.get_config(dataset_name, config_name)
        # Look for a .parquet file path in the config (usually first string value)
        file_path_str = config.get("A_site_sweep") or config.get(
            "hf_path") or config.get("local_subpath")

        # Search for any parquet file in the config
        if not file_path_str:
            for key, val in config.items():
                if isinstance(val, str) and val.endswith('.parquet'):
                    file_path_str = val
                    break

        if file_path_str and file_path_str.endswith('.parquet'):
            # Construct full path
            if loader.mode == 'local' and loader.local_path:
                full_path = loader.local_path / file_path_str
            else:
                # HF mode - download if needed
                cache_root = Path(loader.cache_dir) / \
                    loader.hf_repo.replace("/", "_")
                full_path = cache_root / file_path_str

            if full_path.exists():
                return pd.read_parquet(full_path)
            else:
                raise FileNotFoundError(f"Dataset file not found: {full_path}")

    # Default: load all parquets from the dataset directory
    return loader.load_parquets(dataset_name)


def load_dataset(dataset_name: str, config_name: Optional[str] = None, file_key: str = "A_site_sweep"):
    """
    Intelligently load a dataset using DatasetLoader.
    Handles both simple datasets and nested configurations automatically.

    Args:
        dataset_name: Name of the dataset (e.g., 'MACE_candidates', 'perturbation_experiments')
        config_name: Optional configuration for nested datasets (e.g., 'PKV')
        file_key: For nested configs, which file to load (default 'A_site_sweep')

    Returns:
        pandas DataFrame with the loaded data
    """
    loader = get_loader()

    if config_name:
        # Get the config dictionary for this nested dataset
        config = loader.get_config(dataset_name, config_name)
        # Look for the file path in the config
        file_path = config.get(file_key)
        if file_path and isinstance(file_path, str):
            # Construct the full path
            if loader.mode == 'local':
                full_path = loader.local_path / \
                    file_path if loader.local_path else Path(file_path)
            else:
                # For HF mode, use snapshot_download
                cache_root = Path(loader.cache_dir) / \
                    loader.hf_repo.replace("/", "_")
                full_path = cache_root / file_path

            if full_path.suffix == ".parquet" and full_path.exists():
                return pd.read_parquet(full_path)

    # Default: load all parquets from the dataset directory
    return loader.load_parquets(dataset_name)


def list_datasets() -> Dict[str, str]:
    """Quick listing of all available datasets."""
    return {k: v.get("description", "") for k, v in DATASET_KEY.items() if isinstance(v, dict)}
