"""
model.py — PhysGlobalUNet3D: the original field-conditioned 3D U-Net with switchable global / physics modules.

Phase flags (all new paths are zero-initialized, so each variant starts as its simpler parent):
  attn_blocks > 0   Phase 1  Transformer blocks in the bottleneck (self-attention over the slab, AdaLN-Zero)
  volctx            Phase 2  Volume Context Encoder (2 mm whole volume -> 64 Perceiver tokens) + slab z-position
  phys              Phase 3  'none' | 'concat' (old: physics channels at the input) | 'spade' (3c) |
                             'spade_xattn' (3c + aligned physics cross-attention 3a)
  proto             Phase 3b tissue-prototype tokens from physics maps (joins the context tokens)
  adain_dec         Phase 4  decoder modulation from the global vector
  head              Phase 5  'gate' (original) | 'direct' | 'lut_bias' (intensity LUT x smooth bias + residual)
Conditioning: source field and target field one-hots (5 each) -> MLP.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_FIELDS = 5
PHYS_CH = 5      # [M0, logT1_tgt, logT2_tgt, prior, uncertainty]
PROTO_CH = 5     # [logT1_src, logT2_src, M0, logT1_tgt, logT2_tgt]


def _norm3d(ch):
    return nn.InstanceNorm3d(ch, affine=True)


class ConvBlock3D(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False), _norm3d(cout), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False), _norm3d(cout), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.block(x)


class ResBlock3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReplicationPad3d(1), nn.Conv3d(ch, ch, 3, bias=False), _norm3d(ch), nn.LeakyReLU(0.2, inplace=True),
            nn.ReplicationPad3d(1), nn.Conv3d(ch, ch, 3, bias=False), _norm3d(ch))

    def forward(self, x):
        return x + self.block(x)


class Down3D(nn.Module):
    def __init__(self, cin, cout, pool):
        super().__init__()
        self.conv, self.pool = ConvBlock3D(cin, cout), nn.MaxPool3d(pool, pool)

    def forward(self, x):
        s = self.conv(x)
        return self.pool(s), s


class CBAM3D(nn.Module):
    def __init__(self, ch, reduction=8):
        super().__init__()
        mid = max(ch // reduction, 4)
        self.fc = nn.Sequential(nn.Conv3d(ch, mid, 1, bias=False), nn.ReLU(inplace=True), nn.Conv3d(mid, ch, 1, bias=False))
        self.sa = nn.Conv3d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        w = torch.sigmoid(self.fc(F.adaptive_avg_pool3d(x, 1)) + self.fc(F.adaptive_max_pool3d(x, 1)))
        x = x * w
        s = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return x * torch.sigmoid(self.sa(s))


class UpCBAM3D(nn.Module):
    def __init__(self, cin, cskip, cout, scale):
        super().__init__()
        self.scale, self.cbam, self.conv = scale, CBAM3D(cskip), ConvBlock3D(cin + cskip, cout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.conv(torch.cat([x, self.cbam(skip)], 1))


class AdaIN3D(nn.Module):
    def __init__(self, ch, style_dim):
        super().__init__()
        self.norm, self.fc = nn.InstanceNorm3d(ch, affine=False), nn.Linear(style_dim, 2 * ch)

    def forward(self, x, s):
        g, b = self.fc(s).chunk(2, 1)
        return (1 + g)[..., None, None, None] * self.norm(x) + b[..., None, None, None]


# ----------------------------------------------------------------------------- transformer pieces
def sincos_3d(d, h, w, dim, device):
    """3D sinusoidal positional embedding (d*h*w, dim)."""
    n = dim // 6 * 2
    omega = 1.0 / (10000 ** (torch.arange(n // 2, device=device, dtype=torch.float32) / (n // 2)))
    grids = torch.meshgrid(torch.arange(d, device=device), torch.arange(h, device=device),
                           torch.arange(w, device=device), indexing="ij")
    embs = []
    for g in grids:
        o = g.reshape(-1, 1).float() * omega[None]
        embs += [o.sin(), o.cos()]
    e = torch.cat(embs, 1)
    return F.pad(e, (0, dim - e.shape[1]))


class Attention(nn.Module):
    def __init__(self, dim, heads, kv_dim=None):
        super().__init__()
        kv_dim = kv_dim or dim
        self.h = heads
        self.q, self.kv, self.o = nn.Linear(dim, dim), nn.Linear(kv_dim, 2 * dim), nn.Linear(dim, dim)

    def forward(self, x, ctx=None, bias=None):
        ctx = x if ctx is None else ctx
        B, N, C = x.shape
        q = self.q(x).view(B, N, self.h, C // self.h).transpose(1, 2)
        k, v = self.kv(ctx).view(B, ctx.shape[1], 2, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        mask = None if bias is None else bias[:, None, None, :].to(q.dtype)  # additive per-key bias
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(out.transpose(1, 2).reshape(B, N, C))


class Mlp(nn.Sequential):
    def __init__(self, dim, ratio=4):
        super().__init__(nn.Linear(dim, dim * ratio), nn.GELU(), nn.Linear(dim * ratio, dim))


class TransformerBlock3D(nn.Module):
    """Pre-norm block: AdaLN-Zero self-attention, optional gated cross-attentions (context / physics), MLP."""

    def __init__(self, dim, heads, cond_dim, use_ctx=False, use_phys=False):
        super().__init__()
        self.pe = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)  # convolutional positional encoding
        self.ln1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn, self.mlp = Attention(dim, heads), Mlp(dim)
        self.ada = nn.Linear(cond_dim, 6 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)
        self.use_ctx, self.use_phys = use_ctx, use_phys
        if use_ctx:
            self.ln_c, self.xattn_c = nn.LayerNorm(dim), Attention(dim, heads)
            self.alpha_c = nn.Parameter(torch.zeros(1))
        if use_phys:
            self.ln_p, self.xattn_p = nn.LayerNorm(dim), Attention(dim, heads)
            self.alpha_p = nn.Parameter(torch.zeros(1))

    def forward(self, xmap, cond, ctx=None, phys=None, phys_bias=None, pos=None):
        xmap = xmap + self.pe(xmap)
        B, C, D, H, W = xmap.shape
        x = xmap.flatten(2).transpose(1, 2)
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(cond)[:, None].chunk(6, -1)
        x = x + g1 * self.attn(self.ln1(x) * (1 + sc1) + sh1)
        if self.use_ctx and ctx is not None:
            x = x + torch.tanh(self.alpha_c) * self.xattn_c(self.ln_c(x), ctx)
        if self.use_phys and phys is not None:
            q = self.ln_p(x) + pos
            x = x + torch.tanh(self.alpha_p) * self.xattn_p(q, phys + pos, bias=phys_bias)
        x = x + g2 * self.mlp(self.ln2(x) * (1 + sc2) + sh2)
        return x.transpose(1, 2).reshape(B, C, D, H, W)


class VolumeContextEncoder(nn.Module):
    """2 mm whole volume (B, 1, Z/4, X/4, Y/4) -> n_latents global tokens (Perceiver-style)."""

    def __init__(self, dim=256, n_latents=64, heads=8):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(nn.Conv3d(ci, co, 3, stride=2, padding=1), nn.GroupNorm(8, co), nn.GELU())
        self.cnn = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128), nn.Conv3d(128, dim, 1))
        self.latents = nn.Parameter(torch.randn(n_latents, dim) * 0.02)
        self.ln_l, self.ln_t = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.cross, self.mlp1 = Attention(dim, heads), Mlp(dim)
        self.ln_s, self.self_attn, self.ln_m, self.mlp2 = nn.LayerNorm(dim), Attention(dim, heads), nn.LayerNorm(dim), Mlp(dim)
        self.ln_o = nn.LayerNorm(dim)

    def forward(self, vol):
        f = self.cnn(vol)
        B, C, d, h, w = f.shape
        t = f.flatten(2).transpose(1, 2) + sincos_3d(d, h, w, C, f.device).to(f.dtype)[None]
        lat = self.latents[None].expand(B, -1, -1).to(t.dtype)
        lat = lat + self.cross(self.ln_l(lat), self.ln_t(t))
        lat = lat + self.mlp1(lat)
        lat = lat + self.self_attn(self.ln_s(lat))
        lat = lat + self.mlp2(self.ln_m(lat))
        return self.ln_o(lat)


class TissuePrototypes(nn.Module):
    """Soft-clusters voxels of the 2 mm physics context into K tissue prototypes -> K tokens."""

    def __init__(self, dim=256, k=12, tau=0.02):
        super().__init__()
        # centres in (norm logT1_src, norm logT2_src, M0) space, spread along the T1/T2 diagonal
        c = torch.linspace(0.25, 0.85, k)
        self.centers = nn.Parameter(torch.stack([c, c, torch.full_like(c, 0.7)], 1))
        self.tau = tau
        self.embed = nn.Parameter(torch.randn(k, dim) * 0.02)
        self.proj = nn.Sequential(nn.Linear(PROTO_CH + 1, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, pc, valid):
        """pc (B, 5, n...) physics context; valid (B, 1, n...) mask. Returns tokens (B, K, dim), weights."""
        B = pc.shape[0]
        f = pc.flatten(2).transpose(1, 2)                     # (B, N, 5)
        v = valid.flatten(2).transpose(1, 2).float()          # (B, N, 1)
        d2 = ((f[:, :, None, :3] - self.centers[None, None]) ** 2).sum(-1)  # (B, N, K)
        w = torch.softmax(-d2 / self.tau, -1) * v
        mass = w.sum(1)                                        # (B, K)
        mean = torch.einsum("bnk,bnc->bkc", w, f) / mass[..., None].clamp_min(1e-3)
        frac = (mass / v.sum(1).clamp_min(1.0))[..., None]
        tok = self.proj(torch.cat([mean, torch.log(frac + 1e-4)], -1)) + self.embed[None]
        return tok, w


class SPADE(nn.Module):
    """x * (1 + gamma(phys)) + beta(phys), zero-initialized (identity at start)."""

    def __init__(self, ch, hidden=32):
        super().__init__()
        self.shared = nn.Sequential(nn.Conv3d(PHYS_CH, hidden, 3, padding=1), nn.ReLU(inplace=True))
        self.gb = nn.Conv3d(hidden, 2 * ch, 3, padding=1)
        nn.init.zeros_(self.gb.weight)
        nn.init.zeros_(self.gb.bias)

    def forward(self, x, phys):
        p = F.interpolate(phys, size=x.shape[2:], mode="trilinear", align_corners=False) if phys.shape[2:] != x.shape[2:] else phys
        g, b = self.gb(self.shared(p)).chunk(2, 1)
        return x * (1 + g) + b


class PhysicsEncoder(nn.Module):
    """Physics stack (B, 5, D, H, W) -> tokens on the bottleneck grid, same pooling as the image encoder."""

    def __init__(self, dim, pools):
        super().__init__()
        chs = [16, 32, 64, 128]
        layers, ci = [], PHYS_CH
        for co, p in zip(chs, pools):
            layers += [nn.Conv3d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU(), nn.MaxPool3d(p, p)]
            ci = co
        self.net = nn.Sequential(*layers, nn.Conv3d(ci, dim, 1))

    def forward(self, phys):
        return self.net(phys)


# ----------------------------------------------------------------------------- main model
class PhysGlobalUNet3D(nn.Module):
    def __init__(self, base_ch=32, n_res=2, attn_blocks=0, heads=8, volctx=False, phys="none", proto=False,
                 adain_dec=False, head="gate", n_latents=64, lut_knots=33):
        super().__init__()
        c, dim = base_ch, base_ch * 8
        self.attn_blocks, self.volctx, self.phys, self.proto = attn_blocks, volctx, phys, proto
        self.adain_dec, self.head = adain_dec, head
        self.pools = [(2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2)]
        cond_dim = dim
        self.cond_mlp = nn.Sequential(nn.Linear(2 * N_FIELDS, cond_dim), nn.LeakyReLU(0.2), nn.Linear(cond_dim, cond_dim))
        if volctx:
            self.zpos_mlp = nn.Sequential(nn.Linear(64, cond_dim), nn.LeakyReLU(0.2), nn.Linear(cond_dim, cond_dim))
            self.ctx_enc = VolumeContextEncoder(dim, n_latents, heads)
        if proto:
            self.protos = TissuePrototypes(dim)
            self.type_emb = nn.Parameter(torch.zeros(2, dim))
        in_ch = 1 + (PHYS_CH if phys == "concat" else 0)
        self.d1 = Down3D(in_ch, c, self.pools[0])
        self.d2 = Down3D(c, 2 * c, self.pools[1])
        self.d3 = Down3D(2 * c, 4 * c, self.pools[2])
        self.d4 = Down3D(4 * c, 8 * c, self.pools[3])
        self.bot_conv = ConvBlock3D(8 * c, 8 * c)
        self.res = nn.ModuleList([ResBlock3D(8 * c) for _ in range(n_res)])
        use_ctx = volctx or proto
        use_px = phys == "spade_xattn"
        self.tblocks = nn.ModuleList([TransformerBlock3D(dim, heads, cond_dim, use_ctx, use_px) for _ in range(attn_blocks)])
        if use_px:
            self.phys_enc = PhysicsEncoder(dim, self.pools)
            self.unc_lambda = nn.Parameter(torch.ones(1))
        self.adain = AdaIN3D(8 * c, cond_dim)
        self.u4 = UpCBAM3D(8 * c, 8 * c, 4 * c, (1, 2, 2))
        self.u3 = UpCBAM3D(4 * c, 4 * c, 2 * c, (2, 2, 2))
        self.u2 = UpCBAM3D(2 * c, 2 * c, c, (2, 2, 2))
        self.u1 = UpCBAM3D(c, c, c, (2, 2, 2))
        dec_ch = [4 * c, 2 * c, c, c]
        if phys in ("spade", "spade_xattn"):
            self.spade = nn.ModuleList([SPADE(ch) for ch in dec_ch])
        self.gvec = nn.Sequential(nn.Linear(cond_dim + dim + (dim if use_ctx else 0), dim), nn.LeakyReLU(0.2), nn.Linear(dim, dim))
        if adain_dec:
            self.dec_mod = nn.ModuleList([nn.Linear(dim, 2 * ch) for ch in dec_ch])
            for m in self.dec_mod:
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

        def head_conv(act):
            return nn.Sequential(nn.Conv3d(c, c // 2, 3, padding=1), nn.LeakyReLU(0.2, inplace=True), nn.Conv3d(c // 2, 1, 1), act)
        if head in ("gate", "direct"):
            self.out_synth = head_conv(nn.Sigmoid())
        if head == "gate":
            self.out_gate = head_conv(nn.Sigmoid())
        if head == "lut_bias":
            self.k = lut_knots
            self.lut_fc = nn.Linear(dim, lut_knots - 1)
            nn.init.zeros_(self.lut_fc.weight)
            nn.init.constant_(self.lut_fc.bias, math.log(math.expm1(1.0 / (lut_knots - 1))))  # identity LUT at init
            self.bias_conv = nn.Conv3d(8 * c, 1, 1)
            nn.init.zeros_(self.bias_conv.weight)
            nn.init.zeros_(self.bias_conv.bias)
            self.res_head = nn.Sequential(nn.Conv3d(c, c // 2, 3, padding=1), nn.LeakyReLU(0.2, inplace=True), nn.Conv3d(c // 2, 1, 1))
            nn.init.zeros_(self.res_head[-1].weight)
            nn.init.zeros_(self.res_head[-1].bias)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def zpos_embed(zpos, dim=64):
        freqs = torch.exp(torch.arange(dim // 2, device=zpos.device, dtype=torch.float32) * (-math.log(1000.0) / (dim // 2)))
        a = zpos.float()[:, None] * 1000.0 * freqs[None] / 10.0
        return torch.cat([a.sin(), a.cos()], 1)

    def apply_lut(self, x, knots):
        B = x.shape[0]
        xi = x.clamp(0, 1).reshape(B, -1) * (self.k - 1)
        i0 = xi.floor().clamp(max=self.k - 2)
        f = xi - i0
        i0 = i0.long()
        y = knots.gather(1, i0) * (1 - f) + knots.gather(1, i0 + 1) * f
        return y.view_as(x)

    # ------------------------------------------------------------------ forward
    def forward(self, x, src_idx, tgt_idx, zpos=None, ctx_vol=None, phys=None, phys_ctx=None, proto_in=None):
        """x (B,1,D,H,W); phys (B,5,D,H,W) physics stack; proto_in (pc (B,5,...), valid (B,1,...)) at 2 mm."""
        onehot = torch.cat([F.one_hot(src_idx, N_FIELDS), F.one_hot(tgt_idx, N_FIELDS)], 1).float()
        cond = self.cond_mlp(onehot)
        ctx_tokens, aux = None, {}
        if self.volctx:
            cond = cond + self.zpos_mlp(self.zpos_embed(zpos))
            ctx_tokens = self.ctx_enc(ctx_vol)
        if self.proto:
            ptok, pw = self.protos(*proto_in)
            ptok = ptok + self.type_emb[1]
            ctx_tokens = ptok if ctx_tokens is None else torch.cat([ctx_tokens + self.type_emb[0], ptok], 1)
            aux["proto_mass"] = pw.sum(1)

        net_in = torch.cat([x, phys], 1) if self.phys == "concat" else x
        x1, s1 = self.d1(net_in)
        x2, s2 = self.d2(x1)
        x3, s3 = self.d3(x2)
        x4, s4 = self.d4(x3)
        b = self.bot_conv(x4)
        for r in self.res:
            b = r(b)
        if self.attn_blocks:
            ptoks = pbias = pos = None
            if self.phys == "spade_xattn":
                pm = self.phys_enc(phys)
                ptoks = pm.flatten(2).transpose(1, 2)
                unc = F.adaptive_avg_pool3d(phys[:, 4:5], b.shape[2:]).flatten(1)
                pbias = -F.softplus(self.unc_lambda) * 4.0 * unc
                pos = sincos_3d(*b.shape[2:], b.shape[1], b.device).to(b.dtype)[None]
            for blk in self.tblocks:
                b = blk(b, cond, ctx_tokens, ptoks, pbias, pos)
        b = self.adain(b, cond)

        parts = [cond, b.mean((2, 3, 4))]
        if ctx_tokens is not None:
            parts.append(ctx_tokens.mean(1))
        g = self.gvec(torch.cat(parts, 1))

        u = b
        for i, (up, skip) in enumerate(zip([self.u4, self.u3, self.u2, self.u1], [s4, s3, s2, s1])):
            u = up(u, skip)
            if self.phys in ("spade", "spade_xattn"):
                u = self.spade[i](u, phys)
            if self.adain_dec:
                gm, bm = self.dec_mod[i](g).chunk(2, 1)
                u = u * (1 + gm[..., None, None, None]) + bm[..., None, None, None]

        brain = (x > 0.005).float()
        if self.head == "direct":
            pred = self.out_synth(u)
        elif self.head == "gate":
            gate = self.out_gate(u)
            pred = gate * x + (1 - gate) * self.out_synth(u)
            aux["gate_mean"] = gate.mean()
        else:  # lut_bias
            inc = F.softplus(self.lut_fc(g))
            knots = torch.cat([torch.zeros_like(inc[:, :1]), inc.cumsum(1)], 1)
            log_bias_lr = self.bias_conv(b).clamp(-1.0, 1.0)
            bias = torch.exp(F.interpolate(log_bias_lr, size=x.shape[2:], mode="trilinear", align_corners=False))
            resid = 0.5 * torch.tanh(self.res_head(u))
            pred = self.apply_lut(x, knots) * bias + resid
            aux.update(knots=knots, log_bias_lr=log_bias_lr, resid=resid)
        return (pred * brain).clamp(0.0, 1.0), aux


def build_model(cfg: dict) -> PhysGlobalUNet3D:
    keys = ["base_ch", "n_res", "attn_blocks", "heads", "volctx", "phys", "proto", "adain_dec", "head"]
    return PhysGlobalUNet3D(**{k: cfg[k] for k in keys if k in cfg})
