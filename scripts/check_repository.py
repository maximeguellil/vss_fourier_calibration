"""Validate the tracked collaboration scope without running calibrations."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_NOTEBOOKS = {
    "vss_calibration_CACIB_SP500_SX5E_skew_vars_vols_2y.ipynb",
    "vss_calibration_CACIB_SP500_SX5E_skew_vars_cvc_2y.ipynb",
    "vss_joint_calibration_CACIB_SP500_SX5E_skew_vars_cvc_2y.ipynb",
    "vss_fourier_sanity_check.ipynb",
    "vss_fractional_atm_skew.ipynb",
}

ALLOWED_ROOT_FILES = {
    ".gitattributes",
    ".gitignore",
    ".python-version",
    "README.md",
    "pyproject.toml",
    "uv.lock",
    *{path for path in EXPECTED_NOTEBOOKS if "/" not in path},
}

ALLOWED_PREFIXES = (
    ".github/",
    "scripts/",
    "src/",
    "Plots vss_calibration_CACIB_SP500_SX5E_skew_vars_vols_2y/",
    "Plots vss_calibration_CACIB_SP500_SX5E_skew_vars_cvc_2y/",
    "Plots vss_joint_calibration_CACIB_SP500_SX5E_skew_vars_cvc_2y/",
    "Plots vss_fourier_sanity_check/",
    "Plots vss_fractional_atm_skew/",
)


def tracked_files() -> set[str]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return {
        item.decode("utf-8")
        for item in completed.stdout.split(b"\0")
        if item
    }


def main() -> None:
    tracked = tracked_files()
    unexpected = sorted(
        path
        for path in tracked
        if path not in ALLOWED_ROOT_FILES
        and not path.startswith(ALLOWED_PREFIXES)
    )
    if unexpected:
        raise SystemExit(
            "Tracked files outside the collaboration scope:\n- "
            + "\n- ".join(unexpected)
        )

    forbidden = sorted(
        path
        for path in tracked
        if "/raw/" in f"/{path}/"
        or path.startswith("Market datas/")
        or "10y" in Path(path).name.lower()
        or Path(path).name == ".DS_Store"
        or ".ipynb_checkpoints" in Path(path).parts
    )
    if forbidden:
        raise SystemExit(
            "Forbidden tracked files:\n- " + "\n- ".join(forbidden)
        )

    required = EXPECTED_NOTEBOOKS
    missing = sorted(path for path in required if path not in tracked)
    if missing:
        raise SystemExit(
            "Required collaboration files are not tracked:\n- "
            + "\n- ".join(missing)
        )

    notebook_errors = []
    for path in sorted(path for path in tracked if path.endswith(".ipynb")):
        try:
            notebook = json.loads((ROOT / path).read_text(encoding="utf-8"))
            if not isinstance(notebook.get("cells"), list):
                raise ValueError("missing notebook cell list")
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            notebook_errors.append(f"{path}: {exc}")
    if notebook_errors:
        raise SystemExit(
            "Invalid notebooks:\n- " + "\n- ".join(notebook_errors)
        )

    print(
        f"Repository scope valid: {len(tracked)} tracked files, "
        f"{sum(path.endswith('.ipynb') for path in tracked)} notebooks."
    )


if __name__ == "__main__":
    main()
