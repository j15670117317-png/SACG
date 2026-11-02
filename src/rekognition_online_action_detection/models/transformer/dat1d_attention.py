import torch
import torch.nn as nn
import torch.nn.functional as F


class DAT1DAttention(nn.Module):
    """Standalone 1D DAT-style attention kernel.

    Input/Output contract matches DotProductAttention kernel usage:
    - q, k, v: [B*H, Tq, Dh], [B*H, Tk, Dh], [B*H, Tk, Dh]
    - return: [B*H, Tq, Dh]
    """

    def __init__(self, dropout=0.0, num_heads=8, conv_kernel_size=3):
        super(DAT1DAttention, self).__init__()
        self.dropout = dropout
        self.num_heads = num_heads
        self.conv_kernel_size = conv_kernel_size
        self.last_attn_weights = None

        # These lazy-built layers depend on head_dim (Dh) which is known at runtime.
        self._built = False
        self.dwconv = None
        self.norm = None
        self.channel_gate = None
        self.time_gate = None
        self.proj = None

    def _build_if_needed(self, head_dim, device):
        if self._built:
            return
        padding = self.conv_kernel_size // 2
        self.dwconv = nn.Conv1d(
            head_dim,
            head_dim,
            kernel_size=self.conv_kernel_size,
            stride=1,
            padding=padding,
            groups=head_dim,
            bias=True,
        )
        self.norm = nn.LayerNorm(head_dim)
        self.channel_gate = nn.Sequential(
            nn.Linear(head_dim, max(head_dim // 2, 1)),
            nn.GELU(),
            nn.Linear(max(head_dim // 2, 1), head_dim),
        )
        self.time_gate = nn.Sequential(
            nn.Linear(head_dim, max(head_dim // 2, 1)),
            nn.GELU(),
            nn.Linear(max(head_dim // 2, 1), 1),
        )
        self.proj = nn.Linear(head_dim, head_dim)
        self.to(device)
        self._built = True

    def forward(self, q, k, v, attn_mask=None, knn=False, ratio=0.75):
        b_h, t_q, d_h = q.shape
        t_k = k.shape[1]
        self._build_if_needed(d_h, q.device)

        # Attention branch
        attn_scores = torch.bmm(q, k.transpose(1, 2))
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        if knn:
            keep_k = max(1, int(t_k * ratio))
            sparse_mask = torch.zeros_like(attn_scores)
            index = torch.topk(attn_scores, k=keep_k, dim=-1, largest=True)[1]
            sparse_mask.scatter_(-1, index, 1.0)
            attn_scores = torch.where(
                sparse_mask > 0,
                attn_scores,
                torch.full_like(attn_scores, float("-inf")),
            )

        attn = F.softmax(attn_scores, dim=-1)
        self.last_attn_weights = attn.detach().float().cpu()
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        attn_x = torch.bmm(attn, v)  # [B*H, Tq, Dh]

        # Conv branch on value sequence (local modeling), then align to Tq.
        conv_x = self.dwconv(v.transpose(1, 2)).transpose(1, 2)  # [B*H, Tk, Dh]
        if t_k != t_q:
            conv_x = F.interpolate(
                conv_x.transpose(1, 2),
                size=t_q,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        conv_x = self.norm(conv_x)

        # Cross gating between branches
        ch_map = torch.sigmoid(self.channel_gate(attn_x.mean(dim=1, keepdim=True)))  # [B*H,1,Dh]
        tm_map = torch.sigmoid(self.time_gate(conv_x))  # [B*H,Tq,1]
        fused = attn_x * ch_map + conv_x * tm_map

        fused = self.proj(fused)
        fused = F.dropout(fused, p=self.dropout, training=self.training)
        return fused


if __name__ == "__main__":
    # Quick shape sanity check only; not wired into the project yet.
    bsz, heads, t_q, t_k, d_h = 2, 8, 16, 32, 64
    q = torch.randn(bsz * heads, t_q, d_h)
    k = torch.randn(bsz * heads, t_k, d_h)
    v = torch.randn(bsz * heads, t_k, d_h)
    mod = DAT1DAttention(dropout=0.1, num_heads=heads)
    out = mod(q, k, v)
    print("input q:", q.shape, "k:", k.shape, "v:", v.shape)
    print("output :", out.shape)
