# Validation performed during release preparation

Date: 2026-11-01.

- All eight release YAMLs loaded through YACS 0.1.8 and the supplied original configuration parser, using synthetic metadata strictly for tests.
- Nine unit tests passed, including train/evaluation separation, baseline/SACG config differences, mandatory evaluation checkpoint, path handling, absent data and output preservation.
- All 52 included Python source files parsed successfully. Two pre-existing invalid-escape SyntaxWarnings occur in new_metrics.py docstrings on the preparation Python interpreter.
- All 54 copied original files match their SHA256 provenance manifest.
- The RULSTM utility contains every function imported by main.py.

No PyTorch runtime, real datasets, trained checkpoints or CUDA GPU were used for these checks. Tests of the launcher use synthetic empty files only to exercise existence checks; they are not model tests and are not benchmark data. GPU execution, numerical reproducibility and historical environment compatibility remain to be validated on the actual experiment computer.

Run configuration tests:

```text
python -m pip install -r requirements-check.txt
python -m unittest discover -s tests -v
```
