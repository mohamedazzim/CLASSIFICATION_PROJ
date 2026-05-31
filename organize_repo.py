"""Organize repository: move dataset and notebook into data/ and notebooks/ and copy model into backend models."""
from pathlib import Path
import shutil

cwd = Path.cwd()
# Move CSVs
for csv in cwd.glob('*.csv'):
    dest = cwd / 'data' / csv.name
    shutil.move(str(csv), str(dest))
    print('Moved', csv.name, 'to', dest)
# Move notebooks
for nb in cwd.glob('*.ipynb'):
    dest = cwd / 'notebooks' / nb.name
    shutil.move(str(nb), str(dest))
    print('Moved', nb.name, 'to', dest)
# Move model to backend
src_model = cwd / 'models' / 'employee_attrition_model.pkl'
dest_model = cwd / 'backend' / 'app' / 'models' / 'model.pkl'
if src_model.exists():
    shutil.move(str(src_model), str(dest_model))
    print('Moved model to', dest_model)
else:
    print('Source model not found at', src_model)
