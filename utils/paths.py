import os
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """Finds project root containing 'srccv' or dataset folders."""
    # paths.py is located at <srccv>/utils/paths.py
    # Therefore, srccv_dir is parent.parent, and project root is srccv_dir.parent
    srccv_dir = Path(__file__).resolve().parent.parent
    return srccv_dir.parent


def normalize_sub_dataset_name(name: Optional[str]) -> str:
    """Standardizes sub-dataset name into standard format."""
    if not name:
        return "all"
    clean = str(name).strip()
    low = clean.lower()
    if low in ["all", "combined", "*", "both", "none", "", "tvpreid", "tvpreid_dataset", "tvp-reid", "tvpreid-combined"]:
        return "all"
    if "duke" in low:
        return "TVPReid-Duke"
    if "ilids" in low or "i-lids" in low:
        return "TVPReid-iLIDs"
    if "prid" in low:
        return "TVPReid-PRID"
    if "cuhk" in low or "pedes" in low:
        return "CUHK-PEDES"
    return clean


class PathManager:
    """Manages all relative & absolute paths for SpaceTime-DAMR execution."""
    def __init__(self, custom_data_root: str = "auto", sub_dataset: str = "all"):
        # Explicitly define srccv_dir from file location: utils/paths.py -> utils -> srccv
        # Guarantees ZERO duplicate 'srccv/srccv'
        self.srccv_dir = Path(__file__).resolve().parent.parent
        self.root = self.srccv_dir.parent
        self.sub_dataset = "all"

        # Configure sub-dataset folders for checkpoints & reports
        self.set_sub_dataset(sub_dataset)

        # Determine Data Root
        self.set_data_root(custom_data_root)

    def set_sub_dataset(self, sub_dataset: Optional[str] = "all"):
        """
        Creates and points checkpoints, reports, outputs to dedicated sub-folders.
        Example:
          sub_dataset = 'TVPReid-Duke'
          -> checkpoints_dir = srccv/checkpoints/TVPReid-Duke
          -> reports_dir     = srccv/reports/TVPReid-Duke
        """
        norm_name = normalize_sub_dataset_name(sub_dataset)
        self.sub_dataset = norm_name

        sub_folder = "" if norm_name == "all" else norm_name
        if sub_folder:
            self.checkpoints_dir = self.srccv_dir / "checkpoints" / sub_folder
            self.reports_dir = self.srccv_dir / "reports" / sub_folder
            self.outputs_dir = self.srccv_dir / "outputs" / sub_folder
        else:
            self.checkpoints_dir = self.srccv_dir / "checkpoints" / "all"
            self.reports_dir = self.srccv_dir / "reports" / "all"
            self.outputs_dir = self.srccv_dir / "outputs" / "all"

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def set_data_root(self, custom_data_root: str = "auto"):
        """Resolves data root path prioritizing fast local SSD on Colab."""
        env_data_root = os.environ.get("DATA_ROOT", None)
        if custom_data_root and custom_data_root != "auto" and os.path.exists(custom_data_root):
            self.data_root = Path(custom_data_root)
        elif env_data_root and os.path.exists(env_data_root):
            self.data_root = Path(env_data_root)
        else:
            candidates = [
                Path("/content/dataset"),
                Path("/content/TVPReid_dataset"),
                Path("C:/dataset"),
                self.root / "dataset",
                self.root / "TVPReid_dataset",
                self.srccv_dir / "dataset"
            ]
            self.data_root = self.root / "dataset"
            for cand in candidates:
                if cand.exists():
                    self.data_root = cand
                    break

    def get_checkpoint_path(self, filename: str = "best.pth") -> Path:
        """Returns path to checkpoint in the current sub-dataset folder (with fallback)."""
        target = self.checkpoints_dir / filename
        if target.exists():
            return target

        # Fallback to best_damr.pth if best.pth not found
        if filename == "best.pth" and (self.checkpoints_dir / "best_damr.pth").exists():
            return self.checkpoints_dir / "best_damr.pth"

        # Fallback to root checkpoints dir
        root_ckpt = self.srccv_dir / "checkpoints" / filename
        if root_ckpt.exists():
            return root_ckpt

        return target

    def get_report_path(self, filename: str) -> Path:
        return self.reports_dir / filename


# Global singleton instance
paths = PathManager()
