# Release preparation record — 2026-11-01

Completed:
- Curated supplied core source without altering algorithms or data-loader behavior; hashes in SOURCE_MANIFEST.json.
- Retained the startup RULSTM utility dependency; excluded its unrelated videos/data archives and extra experiments.
- Eight complete YACS configs: baseline/SACG for THUMOS-ResNet, THUMOS-DINOv3, CrossTask and TVSeries-drop_oob.
- Explicit generator attention/CFN settings and disabled unrelated experiment switches.
- Launcher with config/data-path preflight, separate run folders, explicit evaluation checkpoint and run records.
- Environment export utility, English README, data/third-party documentation and release configuration tests.
- Excluded IDE files, Python caches, copied environments, pip bootstrapper, backup scripts, search logs and unrelated data.

Remaining author-supplied items:
- Actual data_info.json and actual held-out validation/training/test IDs.
- Actual successful Windows training environment lock; requirements.txt is only a dependency inventory.
- Checkpoints corresponding to the paper and associated original evaluation/configuration records.
- Final TVSeries extraction/label-policy utilities if claiming to provide full raw-video reproduction.
- Verified publication source data and final class-wise plotting/analysis scripts.
- Upstream license/redistribution confirmation and applicable license files.

The current manuscript specifies CFN scales 0.05 (THUMOS ResNet / TVSeries), 0.08 (THUMOS DINOv3), and 0.03 (CrossTask). These values populate the new configs. Original YAMLs are preserved under docs/original/configs. The new configs are explicit implementation profiles, not proof of historical run identity.

The supplied model files were not refactored to remove other experimental implementations, because doing so could change checkpoint keys or import behavior. Those paths are disabled by the release configs. No private data, credentials or checkpoints were intentionally included; static publication review is still required before external upload.
