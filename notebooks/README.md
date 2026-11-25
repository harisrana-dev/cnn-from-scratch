# Notebooks

Exploratory and explanatory notebooks live here. Each notebook should import
the project code rather than re-implement preprocessing, so experiments stay
consistent with the library.

## Kernel

Register and select the project kernel before running anything:

```powershell
python -m ipykernel install --user --name cnn-from-scratch --display-name "Python (cnn-from-scratch)"
```

Notebooks must start with the repository root on `sys.path` (or import the
package after activating `.venv`):

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / "src"))
```

## Planned notebooks (not yet created)

| Notebook | Phase | Purpose |
|---|---|---|
| `01_dataset_exploration.ipynb` | 1 | Inspect the prepared subsets and figures |
| `02_scratch_cnn_walkthrough.ipynb` | 3-4 | Manual convolution, forward and backward pass |
| `03_keras_cnn_comparison.ipynb` | 5 | Train and evaluate the Keras reference model |

Notebook outputs should be cleared before committing. `.ipynb_checkpoints/` is
git-ignored.
