"""
Multi-scale feature extractors for PPI link prediction:
  - SpectralGraphWavelet  : g(tL) band-pass filters (Hammond-style SGWT)
  - RandomWalkDiffusion   : P^s low-pass diffusion (HyboWaveNet-style)

Both produce multi-scale node features that are concatenated with the
original features and fed into ANY message-passing encoder (GAT/GCN/SAGE/GIN).

The point of having BOTH is a direct, controlled comparison:
    spectral band-pass (ours)  vs  spatial low-pass diffusion (HyboWaveNet).
Same encoder, same decoder, same protocol --- only the multi-scale operator
changes. This isolates the contribution of the *spectral* nature of the
wavelets as opposed to mere multi-scale aggregation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GINConv
except ImportError:
    GATConv = GCNConv = SAGEConv = GINConv = None


# ===========================================================================
#  1. Spectral Graph Wavelet Transform  (OUR method)
#     X_t = g(tL) X,  g(lambda) = lambda * exp(-lambda)   (band-pass)
#     h(L) X,         h(lambda) = exp(-lambda)            (low-pass scaling)
#     Chebyshev approximation, no eigendecomposition.
# ===========================================================================
class SpectralGraphWavelet(nn.Module):
    def __init__(
        self,
        scales=(0.5, 1.0, 2.0, 4.0),
        cheb_order=5,
        lambda_max=2.0,
        num_quad_points=200,
        include_original=True,
    ):
        super().__init__()
        self.scales = list(scales)
        self.cheb_order = cheb_order
        self.lambda_max = lambda_max
        self.num_quad_points = num_quad_points
        self.include_original = include_original

        coeffs = [self._cheb_coefficients(lambda x: torch.exp(-x))]
        for t in self.scales:
            coeffs.append(
                self._cheb_coefficients(lambda x, t=t: (t * x) * torch.exp(-(t * x)))
            )
        coeffs = torch.stack(coeffs, dim=0)
        self.register_buffer("coeffs", coeffs)

        # output multiplier: original + lowpass + J wavelets
        self.out_mult = (1 if include_original else 0) + 1 + len(self.scales)

        # caches (graph fixed during training)
        self._cached_norm = None
        self._cached_key = None
        self._cached_features = None
        self._feat_key = None

    def _cheb_coefficients(self, func):
        K, M = self.cheb_order, self.num_quad_points
        a = self.lambda_max / 2.0
        theta = torch.linspace(0, math.pi, M)
        x = a * (torch.cos(theta) + 1.0)
        fx = func(x)
        cs = []
        for k in range(K + 1):
            cs.append((2.0 / math.pi) * torch.trapz(torch.cos(k * theta) * fx, theta))
        cs = torch.stack(cs)
        cs[0] = cs[0] / 2.0
        return cs

    def _build_norm(self, edge_index, num_nodes, device):
        row, col = edge_index
        deg = torch.bincount(row, minlength=num_nodes).float().to(device)
        dis = deg.pow(-0.5)
        dis[torch.isinf(dis)] = 0.0
        return dis[row] * dis[col]

    def _get_norm(self, edge_index, num_nodes, device):
        key = (id(edge_index), tuple(edge_index.shape), int(num_nodes))
        if self._cached_key != key:
            self._cached_norm = self._build_norm(edge_index, num_nodes, device)
            self._cached_key = key
        return self._cached_norm

    def _matmul_L_tilde(self, x, edge_index, norm):
        row, col = edge_index
        out = torch.zeros_like(x)
        out.index_add_(0, row, norm.unsqueeze(-1) * x[col])
        return -out  # L_tilde = -D^{-1/2} A D^{-1/2} (lambda_max = 2)

    def _apply_filter(self, x, edge_index, norm, coeff):
        K = self.cheb_order
        tx_0 = x
        out = coeff[0] * tx_0
        if K == 0:
            return out
        tx_1 = self._matmul_L_tilde(x, edge_index, norm)
        out = out + coeff[1] * tx_1
        for k in range(2, K + 1):
            tx_2 = 2.0 * self._matmul_L_tilde(tx_1, edge_index, norm) - tx_0
            out = out + coeff[k] * tx_2
            tx_0, tx_1 = tx_1, tx_2
        return out

    @torch.no_grad()
    def _compute(self, x, edge_index):
        num_nodes = x.size(0)
        norm = self._get_norm(edge_index, num_nodes, x.device)
        outputs = []
        if self.include_original:
            outputs.append(x)
        for i in range(self.coeffs.size(0)):
            outputs.append(self._apply_filter(x, edge_index, norm, self.coeffs[i]))
        return torch.cat(outputs, dim=-1)

    def forward(self, x, edge_index):
        key = (id(edge_index), tuple(edge_index.shape), int(x.size(0)))
        if self._feat_key != key:
            self._cached_features = self._compute(x, edge_index)
            self._feat_key = key
        return self._cached_features


# ===========================================================================
#  2. Random Walk Diffusion  (HyboWaveNet-style, spatial low-pass)
#     P = D^{-1} (A + I),   T_s = P^s,   Z = || _s  P^s X
#     This is a LOW-PASS smoothing operator (same family as GCN).
# ===========================================================================
class RandomWalkDiffusion(nn.Module):
    """
    Multi-scale spatial diffusion as in HyboWaveNet [Yu et al., 2025].

    Z = [ X || P^{s_1} X || P^{s_2} X || ... || P^{s_K} X ],
    P = D^{-1}(A + I) is the row-normalized random walk matrix with
    self-loops, and the scales s_k are positive integers (diffusion steps).

    Unlike spectral wavelets g(tL), powers of P act as low-pass filters:
    repeated application of P monotonically smooths the signal, suppressing
    high-frequency components. This is the key conceptual difference we
    isolate in the ablation.
    """

    def __init__(self, scales=(1, 2, 3, 4), include_original=True):
        super().__init__()
        # scales must be positive integers for P^s
        self.scales = [int(s) for s in scales]
        self.include_original = include_original
        self.out_mult = (1 if include_original else 0) + len(self.scales)

        self._cached_norm = None    # row-normalization with self-loops
        self._cached_edges = None   # (row, col) with self-loops added
        self._cached_key = None
        self._cached_features = None
        self._feat_key = None

    def _build(self, edge_index, num_nodes, device):
        row, col = edge_index
        # add self-loops: A_tilde = A + I
        loops = torch.arange(num_nodes, device=device)
        row = torch.cat([row, loops])
        col = torch.cat([col, loops])
        # degree of A_tilde (row-wise)
        deg = torch.bincount(row, minlength=num_nodes).float().to(device)
        deg_inv = deg.pow(-1.0)              # D^{-1} (row-normalized = random walk)
        deg_inv[torch.isinf(deg_inv)] = 0.0
        norm = deg_inv[row]                  # P_{ij} = 1/deg(i) for each edge i->j
        return (row, col), norm

    def _get_operator(self, edge_index, num_nodes, device):
        key = (id(edge_index), tuple(edge_index.shape), int(num_nodes))
        if self._cached_key != key:
            self._cached_edges, self._cached_norm = self._build(edge_index, num_nodes, device)
            self._cached_key = key
        return self._cached_edges, self._cached_norm

    def _matmul_P(self, x, edges, norm):
        row, col = edges
        out = torch.zeros_like(x)
        # (P X)_i = sum_j P_{ij} X_j = sum_{j in N(i)} (1/deg(i)) X_j
        out.index_add_(0, row, norm.unsqueeze(-1) * x[col])
        return out

    @torch.no_grad()
    def _compute(self, x, edge_index):
        num_nodes = x.size(0)
        edges, norm = self._get_operator(edge_index, num_nodes, x.device)

        outputs = []
        if self.include_original:
            outputs.append(x)

        # iteratively compute P^s X, collecting requested scales
        px = x
        max_s = max(self.scales)
        wanted = set(self.scales)
        collected = {}
        for s in range(1, max_s + 1):
            px = self._matmul_P(px, edges, norm)
            if s in wanted:
                collected[s] = px
        for s in self.scales:
            outputs.append(collected[s])
        return torch.cat(outputs, dim=-1)

    def forward(self, x, edge_index):
        key = (id(edge_index), tuple(edge_index.shape), int(x.size(0)))
        if self._feat_key != key:
            self._cached_features = self._compute(x, edge_index)
            self._feat_key = key
        return self._cached_features


# ===========================================================================
#  3. Encoder factory
# ===========================================================================
def build_encoder(name, in_channels, hidden_channels, out_channels,
                  heads=4, dropout=0.3):
    name = name.lower()

    if name == "gat":
        l1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=True)
        l2 = GATConv(hidden_channels * heads, out_channels, heads=1, dropout=dropout, concat=False)
        layers = nn.ModuleList([l1, l2])

        def encode_fn(x, edge_index):
            x = F.elu(layers[0](x, edge_index))
            x = F.dropout(x, p=dropout, training=layers.training)
            return layers[1](x, edge_index)

    elif name == "gcn":
        l1 = GCNConv(in_channels, hidden_channels)
        l2 = GCNConv(hidden_channels, out_channels)
        layers = nn.ModuleList([l1, l2])

        def encode_fn(x, edge_index):
            x = F.relu(layers[0](x, edge_index))
            x = F.dropout(x, p=dropout, training=layers.training)
            return layers[1](x, edge_index)

    elif name == "sage":
        l1 = SAGEConv(in_channels, hidden_channels)
        l2 = SAGEConv(hidden_channels, out_channels)
        layers = nn.ModuleList([l1, l2])

        def encode_fn(x, edge_index):
            x = F.relu(layers[0](x, edge_index))
            x = F.dropout(x, p=dropout, training=layers.training)
            return layers[1](x, edge_index)

    elif name == "gin":
        m1 = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(),
                           nn.Linear(hidden_channels, hidden_channels))
        m2 = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
                           nn.Linear(hidden_channels, out_channels))
        layers = nn.ModuleList([GINConv(m1), GINConv(m2)])

        def encode_fn(x, edge_index):
            x = F.relu(layers[0](x, edge_index))
            x = F.dropout(x, p=dropout, training=layers.training)
            return layers[1](x, edge_index)

    else:
        raise ValueError(f"Unknown encoder: {name}")

    return layers, encode_fn


# ===========================================================================
#  4. Edge decoder
# ===========================================================================
class EdgeMLPDecoder(nn.Module):
    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z, edge_label_index):
        src, dst = edge_label_index
        z_src, z_dst = z[src], z[dst]
        feat = torch.cat([z_src, z_dst, torch.abs(z_src - z_dst), z_src * z_dst], dim=-1)
        return self.mlp(feat).view(-1)


# ===========================================================================
#  5. Unified model for the full comparison matrix
#       feature_mode in {none, proj, spectral, diffusion}
#       encoder_name  in {gat, gcn, sage, gin}
#     This single class produces every variant we need:
#       - baseline (feature_mode='none')
#       - HyboWaveNet-style (feature_mode='diffusion')
#       - ours (feature_mode='spectral')
#     across all four encoders.
# ===========================================================================
class UnifiedGNN(nn.Module):
    def __init__(
        self,
        in_channels,
        encoder_name="gat",
        feature_mode="none",          # 'none' | 'spectral' | 'diffusion'
        hidden_channels=64,
        out_channels=64,
        heads=4,
        dropout=0.3,
        spectral_scales=(0.5, 1.0, 2.0, 4.0),
        diffusion_scales=(1, 2, 3, 4),
        cheb_order=5,
    ):
        super().__init__()
        # The project's arg parser converts the string "none" into Python None.
        # Treat None / empty as the 'none' (no multiscale features) mode.
        if feature_mode is None or str(feature_mode).strip() == "":
            feature_mode = "none"
        feature_mode = str(feature_mode).strip().lower()
        assert feature_mode in ("none", "proj", "spectral", "diffusion"), \
            f"feature_mode must be none/proj/spectral/diffusion, got {repr(feature_mode)}"
        self.feature_mode = feature_mode
        self.dropout = dropout

        if feature_mode == "spectral":
            self.feature_block = SpectralGraphWavelet(
                scales=spectral_scales, cheb_order=cheb_order,
                lambda_max=2.0, include_original=True,
            )
            feat_dim = in_channels * self.feature_block.out_mult
        elif feature_mode == "diffusion":
            self.feature_block = RandomWalkDiffusion(
                scales=diffusion_scales, include_original=True,
            )
            feat_dim = in_channels * self.feature_block.out_mult
        else:
            # 'none' and 'proj': no multi-scale features.
            # 'proj' keeps the same LayerNorm + Linear input block as
            # spectral/diffusion, to separate the effect of multi-scale
            # features from the effect of the extra input layer.
            self.feature_block = None
            feat_dim = in_channels

        if feature_mode != "none":
            self.feat_norm = nn.LayerNorm(feat_dim)
            self.input_proj = nn.Linear(feat_dim, hidden_channels)
            encoder_in = hidden_channels
        else:
            self.feat_norm = None
            self.input_proj = None
            encoder_in = in_channels

        self.encoder_layers, self.encode_fn = build_encoder(
            encoder_name, encoder_in, hidden_channels, out_channels,
            heads=heads, dropout=dropout,
        )
        self.decoder = EdgeMLPDecoder(out_channels, dropout=dropout)

    def encode(self, x, edge_index):
        if self.feature_block is not None:
            x = self.feature_block(x, edge_index)
        if self.feat_norm is not None:
            x = self.feat_norm(x)
            x = F.relu(self.input_proj(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.encode_fn(x, edge_index)

    def decode(self, z, edge_label_index):
        return self.decoder(z, edge_label_index)

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)
