# Geodesic Distance 

# import math
# import torch
# import torch.nn.functional as F
# @torch.no_grad()
# def _build_knn_graph(angle: torch.Tensor, k: int) -> torch.Tensor:
#     """ angle: (K,K) 각(weight), 대칭 행렬. 반환: (K,K) 인접행렬(가중치는 angle, 나머지는 +inf). """
#     K = angle.size(0)
#     inf = torch.full_like(angle, float('inf'))
#     graph = inf.clone()
#     graph[torch.arange(K, device=angle.device), torch.arange(K, device=angle.device)] = 0.0

#     # 각 노드에서 작은 각도 k개 이웃을 양방향 연결
#     # (자기 자신 포함 topk이므로 k+1; 자기자신 제외)
#     topk_idx = torch.topk(-angle, k=min(k + 1, K), dim=1).indices
#     for i in range(K):
#         for j in topk_idx[i]:
#             if i == j:
#                 continue
#             w = angle[i, j]  # 각도 자체가 간선가중치
#             graph[i, j] = w
#             graph[j, i] = w
#     return graph

# def _floyd_warshall(dist_mat: torch.Tensor) -> torch.Tensor:
#     """dist_mat: (K,K), 간선없는 곳은 +inf, 대각 0. torch.minimum 기반으로 미분가능(경로 선택은 비연속)."""
#     d = dist_mat.clone()
#     K = d.size(0)
#     for k in range(K):
#         # 브로드캐스팅으로 벡터화된 relax
#         d = torch.minimum(d, d[:, k:k+1] + d[k:k+1, :])
#     return d

# def geodesic_cross_logits(x: torch.Tensor, y: torch.Tensor,
#                           logit_scale: torch.Tensor,
#                           k: int = 8,
#                           trunc_pi: float = 4.0) -> torch.Tensor:
#     """
#     x: (B,d)  y: (B,d)  (이미 L2-normalize 가정 or 내부에서 정규화)
#     반환: logits_geo (B,B).  Sigmoid 로 들어갈 'logits' 스케일을 유지하기 위해 logit_scale 사용.
#     """
#     B, d = x.size()
#     x_n = F.normalize(x, dim=-1)
#     y_n = F.normalize(y, dim=-1)
#     Z = torch.cat([x_n, y_n], dim=0)            # (2B,d)

#     # 각도 행렬
#     cos = (Z @ Z.t()).clamp(-1.0, 1.0)          # (2B,2B)
#     angle = torch.acos(cos)                      # [0, pi]

#     # kNN 그래프 & 최단경로(지오데식 누적각)
#     G = _build_knn_graph(angle, k=k)             # (2B,2B), inf elsewhere
#     D_geo = _floyd_warshall(G)                   # (2B,2B), 누적각 (라디안)

#     # cross 블록 추출 (x->y)
#     D_xy = D_geo[:B, B:]                         # (B,B)

#     # 누적각 truncate 후 [0,pi]로 선형 매핑 → cosine으로 [-1,1] 유사도
#     # clamp(D, 0, 4π) / 4  ==  D / 4  (D가 4π를 넘으면 4π로 절단)
#     D_xy_clamped = torch.clamp(D_xy, 0.0, trunc_pi * math.pi)
#     theta_norm = D_xy_clamped / (trunc_pi * math.pi) * math.pi  # == D_xy_clamped / trunc_pi
#     sim_geo = torch.cos(theta_norm)              # (B,B) in [-1,1]

#     # logits: 기존 코드와 스케일 일치 유지 (exp(logit_scale) 곱)
#     logits_geo = logit_scale.exp() * sim_geo
#     return logits_geo


import math
import torch
import torch.nn.functional as F

# def geodesic_cross_logits(
#     x: torch.Tensor,                  # (B, d), requires_grad=True
#     y: torch.Tensor,                  # (B, d), requires_grad=True
#     logit_scale: torch.Tensor,        # scalar param (nn.Parameter or Tensor)
#     k: int = 8,
#     trunc_pi: float = 4.0,
#     big: float = 1e6,
#     eps: float = 1e-6,
# ) -> torch.Tensor:
#     """
#     반환: (B,B) logits. 모든 연산은 grad가 x, y 까지 전달되도록 구성.
#     - no_grad 사용 금지
#     - 인덱싱 대입(in-place assign) 금지
#     - where/gather/scatter를 이용해 mask로 선택 (mask 자체는 비연속이지만 값엔 grad 흐름)
#     """
#     B, d = x.shape
#     x_n = F.normalize(x, dim=-1)
#     y_n = F.normalize(y, dim=-1)
#     Z = torch.cat([x_n, y_n], dim=0)                            # (2B, d)

#     # 각도행렬
#     cos = (Z @ Z.t()).clamp(-1.0 + eps, 1.0 - eps)              # 안정성
#     angle = torch.acos(cos)                                     # (2B,2B) in [0, pi]

#     K = angle.size(0)

#     # kNN 마스크(자기자신 포함 topk -> 이후 대각 제거)
#     # topk 자체는 선택 연산(비연속)이지만, 선택된 값(angle.gather)에는 grad가 흐릅니다.
#     _, idx = (-angle).topk(k=min(k + 1, K), dim=1)              # (2B, k+1)
#     mask = torch.zeros_like(angle, dtype=torch.bool)            # (2B,2B)
#     mask = mask.scatter(1, idx, True)                           # True at chosen neighbors
#     mask.fill_diagonal_(False)                                  # 대각 제외 # mask = mask & ~torch.eye(K, dtype=torch.bool, device=mask.device)
#     mask = mask | mask.t()                                      # 대칭

#     # inf 대체 (큰 상수)로 비연결 간선 처리 (min+add에서 안전)
#     inf_mat = torch.full_like(angle, big)
#     graph = torch.where(mask, angle, inf_mat)                   # (2B,2B)
#     graph = graph.clone()
#     graph.fill_diagonal_(0.0)

#     # Floyd–Warshall (torch.minimum + +) : 1차/2차 미분 가능
#     dmat = graph
#     for t in range(K):
#         # 브로드캐스팅 연산; min의 grad는 선택된 가지로 흘러가며 2차 미분도 동작
#         dmat = torch.minimum(dmat, dmat[:, t:t+1] + dmat[t:t+1, :])

#     D_xy = dmat[:B, B:]                                         # (B,B)

#     # 누적각 truncate → [0,pi] 선형 매핑 → cos로 유사도
#     D_xy = torch.clamp(D_xy, 0.0, trunc_pi * math.pi)
#     theta = (D_xy / (trunc_pi * math.pi)) * math.pi
#     sim_geo = torch.cos(theta)                                   # [-1,1]
    
#     # # 1) 포화 비율: clamp에 걸린 비율
#     # with torch.no_grad():
#     #     sat = (D_xy > trunc_pi * math.pi).float().mean().item()
#     #     print('clamp_saturation_ratio:', sat)
#     # # 3) 교차 엣지 비율(마스크에서 x->y가 얼마나 포함되는지)
#     # xy_mask = mask[:B, B:]
#     # print('xy_edges_ratio:', xy_mask.float().mean().item())
    
    
#     logits_geo = logit_scale.exp() * sim_geo
#     return logits_geo


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


# def geodesic_blocks(x, y, k=8, m=2, delta=0.25*math.pi, beta=1.0):
#     """
#     반환: D_xx, D_yy, D_xy, 그리고 exp(-βD) 커널 K_xx, K_yy (필요 시)
#     - 이전 geodesic_cross_logits에서 썼던 kNN + soft-penalty(delta) + Floyd를 그대로 사용
#     - 반드시 no_grad/inf/in-place boolean or/and 금지
#     """
#     # 정규화, angle, kNN 마스크(+ cross diag/top-m), soft penalty delta → Floyd
#     # ... (너가 이미 가진 vectorized Floyd 코드 재사용)
#     # dmat: (2B,2B)
#     B, d = x.shape
#     x = F.normalize(x, dim=-1); y = F.normalize(y, dim=-1)
#     Z = torch.cat([x, y], 0)

#     cos = (Z @ Z.t()).clamp(-1+1e-6, 1-1e-6)
#     angle = torch.acos(cos)                             # (2B,2B)

#     K = 2*B
#     _, idx = (-angle).topk(k=min(k+1, K), dim=1)
#     mask = torch.zeros_like(angle, dtype=torch.bool).scatter(1, idx, True)
#     mask.fill_diagonal_(False)
#     mask = mask | mask.t()

#     # B: cross-diagonal edges + cross top-m
#     diag = torch.arange(B, device=angle.device)
#     mask[diag, B+diag] = True
#     mask[B+diag, diag] = True
#     if m > 0:
#         _, idx_xy = (-angle[:B, B:]).topk(k=min(m, B), dim=1)
#         mask[:B, B:].scatter_(1, idx_xy, True)
#         mask[B:, :B] = mask[:B, B:].t()

#     # C: non-neighbors get small penalty, not infinity
#     graph = torch.where(mask, angle, angle + delta)
#     graph = graph.clone(); graph.fill_diagonal_(0.0)

#     dmat = graph
#     for t in range(K):
#         dmat = torch.minimum(dmat, dmat[:, t:t+1] + dmat[t:t+1, :])
        
#     B = x.size(0)
#     D_xx = dmat[:B, :B]
#     D_xy = dmat[:B, B:]
#     D_yy = dmat[B:, B:]
#     K_xx = torch.exp(-beta * D_xx)
#     K_yy = torch.exp(-beta * D_yy)
#     return D_xx, D_yy, D_xy, K_xx, K_yy

def geodesic_blocks_simple(x, y, k=8, m_cross=2, delta=0.25*math.pi):
    # 정규화
    x = F.normalize(x, dim=-1); y = F.normalize(y, dim=-1)
    Z = torch.cat([x, y], 0)  # (2B,d)
    B = x.size(0); K = 2*B

    cos = (Z @ Z.t()).clamp(-1+1e-6, 1-1e-6)
    angle = torch.acos(cos)  # (2B,2B)

    # kNN 마스크 (자기 제외 + 대칭)
    _, idx = (-angle).topk(k=min(k+1, K), dim=1)
    mask = torch.zeros_like(angle, dtype=torch.bool).scatter(1, idx, True)
    mask.fill_diagonal_(False)
    mask = mask | mask.t()

    # cross 보강: 대각 + cross top-m
    diag = torch.arange(B, device=angle.device)
    mask[diag, B+diag] = True; mask[B+diag, diag] = True
    if m_cross > 0:
        _, idx_xy = (-angle[:B, B:]).topk(k=min(m_cross, B), dim=1)
        mask[:B, B:].scatter_(1, idx_xy, True)
        mask[B:, :B] = mask[:B, B:].t()

    # 소프트 페널티 delta (inf 대신) -> 경로 끊김/포화 방지
    graph = torch.where(mask, angle, angle + delta)
    graph = graph.clone(); graph.fill_diagonal_(0.0)

    # Floyd–Warshall
    dmat = graph
    for t in range(K):
        dmat = torch.minimum(dmat, dmat[:, t:t+1] + dmat[t:t+1, :])

    D_xx = dmat[:B, :B]
    D_xy = dmat[:B, B:]
    D_yy = dmat[B:, B:]
    return D_xx, D_yy, D_xy