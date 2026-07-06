import math
import torch
import torch.nn.functional as F

def geodesic_cross_logits(x, y, logit_scale, k=8, m=2, delta=0.25*math.pi,  # C
                          use_exp_kernel=True, gamma=1.0):                  # A
    B, d = x.shape
    x = F.normalize(x, dim=-1); y = F.normalize(y, dim=-1)
    Z = torch.cat([x, y], 0)

    cos = (Z @ Z.t()).clamp(-1+1e-6, 1-1e-6)
    angle = torch.acos(cos)                             # (2B,2B)

    K = 2*B
    _, idx = (-angle).topk(k=min(k+1, K), dim=1)
    mask = torch.zeros_like(angle, dtype=torch.bool).scatter(1, idx, True)
    mask.fill_diagonal_(False)
    mask = mask | mask.t()

    # B: cross-diagonal edges + cross top-m
    diag = torch.arange(B, device=angle.device)
    mask[diag, B+diag] = True
    mask[B+diag, diag] = True
    if m > 0:
        _, idx_xy = (-angle[:B, B:]).topk(k=min(m, B), dim=1)
        mask[:B, B:].scatter_(1, idx_xy, True)
        mask[B:, :B] = mask[:B, B:].t()

    # C: non-neighbors get small penalty, not infinity
    graph = torch.where(mask, angle, angle + delta)
    graph = graph.clone(); graph.fill_diagonal_(0.0)

    dmat = graph
    for t in range(K):
        dmat = torch.minimum(dmat, dmat[:, t:t+1] + dmat[t:t+1, :])

    D_xy = dmat[:B, B:]                                # (B,B)

    # A: no clamp; everywhere-differentiable similarity
    if use_exp_kernel:
        sim_geo = torch.exp(-gamma * D_xy)             # (B,B) in (0,1]
    else:
        Dn = D_xy / (D_xy.detach().max() + 1e-6)
        sim_geo = 1.0 - Dn

    return logit_scale.exp() * sim_geo

def geodesic_blocks_simple(x, y, k=8, m_cross=2, delta=0.25*math.pi):
    
    x = F.normalize(x, dim=-1); y = F.normalize(y, dim=-1)
    Z = torch.cat([x, y], 0)  # (2B,d)
    B = x.size(0); K = 2*B

    cos = (Z @ Z.t()).clamp(-1+1e-6, 1-1e-6)
    angle = torch.acos(cos)  # (2B,2B)

    _, idx = (-angle).topk(k=min(k+1, K), dim=1)
    mask = torch.zeros_like(angle, dtype=torch.bool).scatter(1, idx, True)
    mask.fill_diagonal_(False)
    mask = mask | mask.t()

    diag = torch.arange(B, device=angle.device)
    mask[diag, B+diag] = True; mask[B+diag, diag] = True
    if m_cross > 0:
        _, idx_xy = (-angle[:B, B:]).topk(k=min(m_cross, B), dim=1)
        mask[:B, B:].scatter_(1, idx_xy, True)
        mask[B:, :B] = mask[:B, B:].t()

    graph = torch.where(mask, angle, angle + delta)
    graph = graph.clone(); graph.fill_diagonal_(0.0)

    dmat = graph
    for t in range(K):
        dmat = torch.minimum(dmat, dmat[:, t:t+1] + dmat[t:t+1, :])

    D_xx = dmat[:B, :B]
    D_xy = dmat[:B, B:]
    D_yy = dmat[B:, B:]
    return D_xx, D_yy, D_xy
