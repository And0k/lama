Do not break existing configs not related to the netcdf pipeline: add new YAML to configs/ instead of editing the original ones, so that you can roll back. Don't keep backward compatibility of netcdf pipeline: refactor for minimum code, maximum clarity, speed, essential functionality.

logging with correct level over print(), logging.exception() on errors
No hardcoded values in log messages — uses %s with variable names
No hardcoded values in comments if not necessary to show example of calculations

Hydra and _target_: if you use _target_, make sure the path is correct and the module is imported (check python -c "import <module>; print('ok')").

num_workers and netCDF: if there are errors, open netCDF inside __getitem__ or use worker_init_fn.

to run python use `source .venv/bin/activate`

Tests: always first batch_size=1, max_epochs=1.


Document configuration changes in configs/README.md so others can replicate.
Keep README_nc.md about netcdf pipeline up to date