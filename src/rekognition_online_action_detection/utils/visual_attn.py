import os
import os.path as osp

import numpy as np
import torch


def _feature_to_rgb_grid(feats_btC: torch.Tensor):
    """Project per-timestep feature vector to a pseudo-RGB heatmap image.

    feats: [B, T, C] on any device; returns float32 numpy [H, W, 3] in [0, 1].
    """
    feats = feats_btC.detach().float().cpu()
    if feats.ndim != 3:
        raise ValueError('visual_inputs must be [B, T, C], got shape {}'.format(tuple(feats.shape)))
    _b, tlen, c = feats.shape
    if tlen <= 0 or c <= 0:
        raise ValueError('empty visual feature tensor: T={}, C={}'.format(tlen, c))

    # simple PCA-like projection via random fixed directions (deterministic per call)
    g = torch.Generator(device='cpu')
    g.manual_seed(1337)
    proj = torch.randn(c, 3, generator=g)
    proj = proj / (proj.norm(dim=0, keepdim=True) + 1e-6)
    rgb = torch.matmul(feats[0], proj)  # [T, 3]
    rgb = rgb - rgb.min(dim=0, keepdim=True).values
    rgb = rgb / (rgb.max(dim=0, keepdim=True).values + 1e-6)

    # layout as a wide strip: height=max(8, sqrt(T)), width=ceil(T/H)
    tlen = int(rgb.shape[0])
    h = int(max(8, int(np.sqrt(tlen))))
    w_ = int(np.ceil(tlen / float(h)))
    pad = h * w_ - tlen
    if pad > 0:
        rgb = torch.cat([rgb, rgb[-1:].repeat(pad, 1)], dim=0)
    img = rgb.view(h, w_, 3).clamp(0, 1).numpy()
    return img


def dump_decoder_attention_png(
    cfg,
    epoch,
    batch_idx,
    sample_idx,
    attn_kind,
    layer_idx,
    attn_map_btqq,
    visual_inputs_btC: torch.Tensor,
    out_dir,
    logger,
    seg=None,
):
    """Save one PNG with attention heatmap + pseudo-RGB feature strip.

    seg (optional): dict with integer lengths for overlay guides:
      - tgt: [ctx, work, ant] lengths for QUERY axis (sum should match attn rows)
      - mem: [long, ctx, work, ant] lengths for KEY axis for CMeRT+include_work:
             keys = concat(long_memory, work_memories) where work_memories includes ctx/work/ant
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        logger.warning('[vis] matplotlib not available; skip attn dump ({})'.format(e))
        return

    os.makedirs(out_dir, exist_ok=True)
    fname = 'attn_ep{:03d}_b{:05d}_s{}_{}_L{}.png'.format(
        int(epoch), int(batch_idx), int(sample_idx), str(attn_kind), int(layer_idx))
    path = osp.join(out_dir, fname)

    attn = attn_map_btqq[sample_idx].detach().float().cpu().numpy()
    if attn.size == 0 or attn.ndim != 2:
        logger.warning('[vis] skip attn dump: empty or bad attn shape {}'.format(getattr(attn, 'shape', None)))
        return
    attn = attn - attn.min()
    attn = attn / (attn.max() + 1e-6)

    rgb = _feature_to_rgb_grid(visual_inputs_btC[sample_idx:sample_idx + 1])

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 2, 1)
    im_attn = ax.imshow(attn, cmap='viridis', aspect='auto', vmin=0.0, vmax=1.0)
    fig.colorbar(im_attn, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('Attention (mean heads)')
    ax.set_xlabel('key index (memory tokens)')
    ax.set_ylabel('query index (decoder tgt tokens)')

    n_q, n_k = attn.shape[0], attn.shape[1]
    if seg is not None:
        mem_parts = seg.get('mem')
        tgt_parts = seg.get('tgt')
        if isinstance(mem_parts, (list, tuple)) and len(mem_parts) == 4:
            long_n, ctx_k, work_k, ant_k = [int(x) for x in mem_parts]
            # Expected key layout: long | ctx | work | ant
            xs = [0]
            acc = 0
            for name, ln in [('long', long_n), ('ctx', ctx_k), ('work', work_k), ('ant', ant_k)]:
                acc += max(ln, 0)
                xs.append(acc)
            if xs[-1] != n_k:
                # This happens in ablations (e.g., anticipation=0, include_work toggles)
                # where the captured attention's key length doesn't match the assumed layout.
                # Skip overlay rather than spamming warnings.
                logger.warning(
                    '[vis] attn key len mismatch: expected_sum={} actual={} (skip overlay)'.format(
                        xs[-1], n_k
                    )
                )
            else:
                # draw dividers between segments (skip 0 and end)
                colors = {
                    'long|ctx': 'w',
                    'ctx|work': 'c',
                    'work|ant': 'm',
                }
                labels = [('long|ctx', xs[1]), ('ctx|work', xs[2])]
                if ant_k > 0:
                    labels.append(('work|ant', xs[3]))
                for lab, x in labels:
                    if 0 < x < n_k:
                        ax.axvline(x - 0.5, color=colors.get(lab, 'w'), linewidth=1.2, alpha=0.9)
                        ax.text(
                            x - 0.5,
                            0.5,
                            lab,
                            color='w',
                            fontsize=8,
                            ha='center',
                            va='top',
                            alpha=0.95,
                        )

        if isinstance(tgt_parts, (list, tuple)) and len(tgt_parts) == 3:
            ctx_q, work_q, ant_q = [int(x) for x in tgt_parts]
            ys = [0, ctx_q, ctx_q + work_q, ctx_q + work_q + ant_q]
            if ys[-1] != n_q:
                logger.warning(
                    '[vis] attn query len mismatch: expected_sum={} actual={} (skip overlay)'.format(
                        ys[-1], n_q
                    )
                )
            else:
                labels = [('ctx|work', ys[1])]
                if ant_q > 0:
                    labels.append(('work|ant', ys[2]))
                for lab, y in labels:
                    if 0 < y < n_q:
                        ax.axhline(y - 0.5, color='w', linewidth=1.0, alpha=0.85)
                        ax.text(
                            0.5,
                            y - 0.5,
                            lab,
                            color='w',
                            fontsize=8,
                            ha='left',
                            va='center',
                            alpha=0.95,
                        )

    ax_rgb = fig.add_subplot(1, 2, 2)
    ax_rgb.imshow(rgb, aspect='auto')
    ax_rgb.set_title('Visual feature strip (pseudo-RGB)')
    ax_rgb.set_xlabel('time bins')
    ax_rgb.set_ylabel('')

    fig.suptitle(
        'epoch {} | batch {} | sample {} | {} L{} | queries=tgt(context+work+ant)'.format(
            epoch, batch_idx, sample_idx, str(attn_kind), int(layer_idx)
        )
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    logger.info('[vis] wrote attention png: {}'.format(path))
