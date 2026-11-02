# SACG: Selective Attention and Contrastive Generator

Code for **Selective memory retrieval and contrastive normalization for online action detection**.

**Authors:** Pu Han, Cenzan Jing (corresponding author), and Xu Gao.  
**Contact:** jcz@gs.zzu.edu.cn

SACG modifies the future generator of CMeRT. Adaptive Sparse Temporal Attention (ASTA, called `assa` in the code) replaces generator cross-attention. Contrastive Feature Normalization (CFN, called `GEN_CONTRANORM`) is applied after the cross-attention residual and LayerNorm. Generator self-attention and the surrounding CMeRT modules retain dot-product attention.

## Release status

This is a curated source package, not a self-contained data/weights release. The supplied model, loader, losses, metrics and `main.py` are preserved byte-for-byte; `SOURCE_MANIFEST.json` records their hashes. Publication configurations and a launcher were added. No benchmark scores were recomputed while preparing this package.

**Before an experiment can run, provide your actual `data/data_info.json`, features and processed targets.** They were not present in the supplied source folder. Model checkpoints, the actual Windows environment lock, the held-out hyperparameter-selection split and the final TVSeries preprocessing scripts also remain to be added. See [RELEASE_STATUS.md](docs/RELEASE_STATUS.md). Do not treat a missing split or checkpoint as a value to guess.

The configs encode the current manuscript settings; they are not recovered historical per-run configurations and do not independently establish checkpoint provenance.

## 1. Environment

Use the Python/PyTorch environment that successfully ran the experiments. Do not copy a `venv` directory or substitute the WSL feature-extraction environment for the Windows training environment.

- Core code imports PyTorch, NumPy, pandas, SciPy, scikit-learn, einops, tqdm, YACS and Matplotlib.
- Install a PyTorch build compatible with the target operating system/GPU before installing the remaining requirements.
- `requirements.txt` lists import dependencies, **not an exact historical version lock**. The original dependency file is preserved in `docs/original/requirements.txt` for provenance.

```powershell
python -m pip install -r requirements.txt
```

For config-only checks on a machine without PyTorch:

```powershell
python -m pip install -r requirements-check.txt
python scripts/run_experiment.py plan
python -m unittest discover -s tests -v
```

Capture the actual environment on the experiment computer before archival:

```powershell
python scripts/export_environment.py --output environment_actual
```

The launcher uses the active `python`; it does not activate Conda or change your environment automatically.

## 2. Data layout

Copy the **actual experiment metadata** into `data/data_info.json` (or supply `--data-info`). Dataset keys are exactly `THUMOS`, `CrossTask` and `TVSeries`. See [data/README.md](data/README.md) for required fields and split conventions. No fabricated metadata is included.

```text
data/
  data_info.json
  THUMOS/
    rgb_kinetics_resnet50/<video_id>.npy
    flow_kinetics_bninception/<video_id>.npy
    target_perframe/<video_id>.npy
    dinov3_vit7b16_rgb_fps24_len6_pos4_stride6_h448_w448/<video_id>.npy
  CrossTask/
    rgb_i3d/<video_id>.npy
    rgb_resnet152_audio/<video_id>.npy
    target_perframe/<video_id>.npy
  TVSeries/
    rgb_kinetics_resnet50/<episode_id>.npy
    flow_kinetics_bninception/<episode_id>.npy
    target_perframe_drop_oob/<episode_id>.npy
```

Use `--data-root` for a dataset stored elsewhere. This overrides the config's default `data/<dataset>` location. It does not copy, crop, resample or relabel data.

Original data are obtained through the respective providers: [THUMOS'14](https://www.crcv.ucf.edu/THUMOS14/), [CrossTask](https://github.com/DmZhukov/CrossTask), and [TVSeries](https://homes.esat.kuleuven.be/psi-archive/rdegeest/TVSeries.html). See the [CMeRT repository](https://github.com/pangzhan27/CMeRT) for the upstream feature preparation references. TVSeries video and feature files are not included in this release.

## 3. Explicit baseline and SACG configurations

Each configuration is complete; no `_BASE_` inheritance or extra SACG command-line switches are needed.

| Setting | Baseline config | SACG config | SACG CFN scale | Metric | Manuscript reference checkpoint |
|---|---|---|---:|---|---|
| THUMOS, ResNet-50 | `configs/THUMOS/baseline_resnet50.yaml` | `configs/THUMOS/sacg_resnet50.yaml` | 0.05 | mAP | epoch 9 |
| THUMOS, DINOv3 | `configs/THUMOS/baseline_dinov3.yaml` | `configs/THUMOS/sacg_dinov3.yaml` | 0.08 | mAP | epoch 7 |
| CrossTask | `configs/CrossTask/baseline_i3d.yaml` | `configs/CrossTask/sacg_i3d.yaml` | 0.03 | mAP | epoch 9 |
| TVSeries, drop_oob | `configs/TVSeries/baseline_drop_oob.yaml` | `configs/TVSeries/sacg_drop_oob.yaml` | 0.05 | mcAP | supply the actual checkpoint; no epoch inferred |

The last column documents the current manuscript, not an automatic best-epoch selector. Evaluation always requires an explicit checkpoint path. DINOv3 uses 4,096-dimensional RGB features; only its RGB input changes relative to the ResNet THUMOS setting, in addition to the stated CFN scale.

All SACG configs explicitly specify:

```yaml
GEN_SELF_ATTN_TYPE: dotproduct
GEN_CROSS_ATTN_TYPE: assa
ASSA_TAU: 1.0
GEN_CROSS_ASSA_INIT_W: [0.0, 0.0]
GEN_CONTRANORM: true
GEN_CONTRANORM_POS: cross
GEN_CONTRANORM_TEMP: 1.0
```

The equal initial logits correspond to sparse-branch proportion `p_s = 0.50`. CFN dual normalization, learnable scaling, identity and pre-normalization options are disabled. Other attention sites use dot-product attention, and MQ-MoE, ASSA-v2, attention residual experiments and FC normalization are not enabled by these configs. Baseline configs set `GEN_CROSS_ATTN_TYPE: dotproduct`, `GEN_CONTRANORM: false` **and** `GEN_CONTRANORM_POS: none`.

Internal names `CMeRT`, `assa` and `ContraNorm1D` are retained for checkpoint compatibility. Selecting `MODEL_NAME: CMeRT` does not by itself determine whether the generator is baseline or SACG; the explicit switches do.

## 4. Recommended launcher (Windows PowerShell or Linux/WSL)

Run from the extracted `SACG` folder. The following one-line commands work in both shells and avoid undefined PowerShell variables.

List all profiles, without data or a GPU:

```powershell
python scripts/run_experiment.py plan
```

After supplying actual metadata and arrays, check a configuration:

```powershell
python scripts/run_experiment.py check --config configs/THUMOS/sacg_resnet50.yaml
```

This validates YACS config loading, required metadata, disjoint train/evaluation lists and the existence of all feature/target paths. It does **not** check numerical array contents, temporal alignment or the historical label-generation process.

Train separate seed-0 baseline and SACG runs:

```powershell
python -u scripts/run_experiment.py train --config configs/THUMOS/baseline_resnet50.yaml --seed 0
python -u scripts/run_experiment.py train --config configs/THUMOS/sacg_resnet50.yaml --seed 0
```

Evaluate each corresponding epoch-9 checkpoint:

```powershell
python -u scripts/run_experiment.py eval --config configs/THUMOS/baseline_resnet50.yaml --seed 0 --checkpoint outputs/THUMOS/baseline_resnet50/seed0/epoch-9.pth
python -u scripts/run_experiment.py eval --config configs/THUMOS/sacg_resnet50.yaml --seed 0 --checkpoint outputs/THUMOS/sacg_resnet50/seed0/epoch-9.pth
```

For CrossTask/DINOv3, replace the config and use the corresponding explicit checkpoint. For TVSeries on the author's F drive, for example:

```powershell
python scripts/run_experiment.py check --config configs/TVSeries/sacg_drop_oob.yaml --data-info "D:/CMeRT-main/data/data_info.json" --data-root "F:/TVSeries_MAT_dual_v2_20260919"
python -u scripts/run_experiment.py train --config configs/TVSeries/sacg_drop_oob.yaml --data-info "D:/CMeRT-main/data/data_info.json" --data-root "F:/TVSeries_MAT_dual_v2_20260919" --seed 0
```

These example paths must exist on your machine. Under WSL, use `/mnt/d/...` and `/mnt/f/...` instead. The metadata file must contain the correct TVSeries entry.

Options: `--gpu 0`, `--workers 8`, `--seed 1`, `--output PATH`, `--session NAME`. The launcher preserves the config's batch size and other training parameters. Each run records the resolved config, full argument list, installed package versions and child-process exit status. Training logs and checkpoints continue to be written by the original code.

**Existing output folders are never reused by the launcher.** Choose a different `--session` for a deliberate new run or repeated evaluation. This launcher starts fresh training; it is not a crash-resume or queue scheduler. A failed run is retained for diagnosis. It does not launch experiments merely by extracting this archive.

## 5. Direct main.py command (PowerShell)

The existing interface is unchanged. Unlike the launcher, it does not enforce unique output directories or save the additional release records.

```powershell
python -u main.py `
  --config_file configs/THUMOS/sacg_resnet50.yaml `
  --gpu 0 `
  OUTPUT_DIR outputs `
  SESSION sacg_seed0 `
  SEED 0
```

```powershell
python -u main.py `
  --test 1 `
  --config_file configs/THUMOS/sacg_resnet50.yaml `
  --gpu 0 `
  OUTPUT_DIR outputs_eval `
  SESSION sacg_seed0 `
  SEED 0 `
  MODEL.CHECKPOINT outputs/THUMOS/sacg_resnet50/sacg_seed0/epoch-9.pth
```

Notice that the direct example uses session `sacg_seed0`, whereas the launcher defaults to `seed0`. Do not mix their checkpoint paths. In PowerShell, the backtick must be the final character on a continued line. For baseline, use its baseline config and its own checkpoint. Never evaluate a SACG checkpoint using a baseline config.

## 6. Protocol details to preserve

- **TVSeries:** both baseline and SACG profiles use `target_perframe_drop_oob`, 31 target columns, 4 FPS and `cAP`. Only the five completely out-of-bounds Punch intervals in `How_I_Met_Your_Mother_ep4` are omitted in the reported policy. This is an experimental label policy, not an author-confirmed dataset correction. The `punch_minus300` alternative is not substituted by these profiles.
- **THUMOS:** the supplied loader retains both `video_test_0000270` and `video_test_0001496`; the exclusion block in the parser is disabled. The historical 604 RGB/target vs. 1,206 flow-row case is not repaired or resampled by this release. Original indexing is retained; that alone does not verify temporal alignment.
- **Validation selection:** the actual held-out subset within the 200 official validation videos must be published. The original training loop names its evaluation phase `test`; when conducting parameter selection, supply a metadata file whose `test_session_set` contains the held-out validation IDs. Use separate final-evaluation metadata for official test videos, and document the actual final training set. Do not infer or silently change the historical splits.
- **Checkpoint loading:** the original loader reads `model_state_dict` strictly. It also stores `optimizer_state_dict` and `epoch`. The launcher does not guess EMA/raw-weight provenance or select checkpoints by test performance.
- `batch` inference uses the original online temporal window logic; it is not permission to use future observed frames.

## 7. Diagnostics and attribution

The original diagnostic scripts are included at their original paths: `gradient_diagnostics.py`, `measure_th14_efficiency.py`, `plot_figure2_loss_curve.py`, `plot_figure4_contranorm_similarity.py` and `tools/`. Their historical figure numbers may differ from the current manuscript. Verify the source data and checkpoint before using them to regenerate a publication figure. Unverified historical CSVs and large search logs are not bundled as final results.

This implementation builds on [CMeRT](https://github.com/pangzhan27/CMeRT), which credits [TeSTra](https://github.com/zhaoyue-zephyrus/TeSTra) and [MAT](https://github.com/Echo0125/MAT-Memory-and-Anticipation-Transformer). The retained RULSTM utility is identified in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). See the manuscript references for the attention and normalization methods. Preserve upstream attribution and copyright notices.

**No blanket license has been assigned to third-party code.** Resolve upstream redistribution/license details before public archival; see [LICENSE_STATUS.md](LICENSE_STATUS.md). No repository DOI or accepted-publication metadata is invented in this draft.
