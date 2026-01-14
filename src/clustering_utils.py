import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gc

from fast_pytorch_kmeans import KMeans

def l2_normalize(x, dim=-1, eps=1e-12):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def pca_2d_fit(x: torch.Tensor):
    """x( [N,D] )에 대해 한 번만 PCA basis를 학습하고 2D로 투영해서 반환."""
    mean = x.mean(dim=0, keepdim=True)
    x0 = x - mean
    U, S, V = torch.pca_lowrank(x0, q=2)
    Z = x0 @ V[:, :2]  # [N,2]
    return Z, V, mean  # 좌표, basis, 평균

class KMeansCluster:
    def __init__(self, img_feats: torch.Tensor, device=None, mode: str = "cosine"):
        """
        img_feats: [N, D] torch.Tensor (cpu/gpu 상관없음)
        """
        self.img_feats = img_feats
        self.device = device or (img_feats.device if isinstance(img_feats, torch.Tensor) else 'cpu')
        # 저장용 버퍼
        self._labels = None          # [N]
        self._centers = None         # [k, D]
        self._center_assign = None   # 각 center가 가리키는 인덱스 [k]
        self.mode = mode
    
    def free(self):
        """Release ALL GPU tensors / resources held by this instance."""
        # Drop PyTorch CUDA tensors
        for name, val in list(self.__dict__.items()):
            if torch.is_tensor(val) and val.is_cuda:
                setattr(self, name, None)
        self.img_feats = None
        self._labels = None          # [N]
        self._centers = None         # [k, D]
        self._center_assign = None   # 각 center가 가리키는 인덱스 [k]

        # Force collection
        gc.collect()
        torch.cuda.empty_cache()
    
    # Nice-to-have: context manager sugar
    def __enter__(self):
        return self
    def __exit__(self, et, ev, tb):
        self.free()

    def euclidean_dist(self, x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()
        return dist

    def query(self, n):
        """
        n개 클러스터로 kmeans → 각 클러스터 중심에 가장 가까운 실제 샘플 인덱스 반환.
        시각화를 위해 labels/centers도 내부에 저장.
        """
        embeddings = self.img_feats
        index = torch.arange(len(embeddings), device=embeddings.device)

        kmeans = KMeans(n_clusters=n, mode=self.mode, verbose=False)
        labels = kmeans.fit_predict(embeddings)
        centers = kmeans.centroids  # [n, D]

        dist_matrix = self.euclidean_dist(centers, embeddings)  # [n, N]
        q_idxs = index[torch.argmin(dist_matrix, dim=1)]        # [n]

        # 저장 (시각화 시 사용)
        self._labels = labels.detach()
        self._centers = centers.detach()
        self._center_assign = q_idxs.detach()

        return q_idxs

    def query_incluster(self, n):
        """
        n개 클러스터로 kmeans → 각 클러스터 중심에서
        해당 클러스터에 속한 샘플 중 가장 가까운 인덱스를 반환.
        """
        embeddings = self.img_feats
        index = torch.arange(len(embeddings), device=embeddings.device)

        # k-means 학습
        kmeans = KMeans(n_clusters=n, mode=self.mode, verbose=False)        # euclidean으로 되어 있었음 (v11까지)
        labels = kmeans.fit_predict(embeddings)
        centers = kmeans.centroids  # [n, D]

        q_idxs = []
        for c in range(n):
            mask = (labels == c)
            if mask.sum() == 0:
                continue
            pts = embeddings[mask]
            ids = index[mask]

            if self.mode == "cosine":
                p = F.normalize(pts, dim=1)
                cen = F.normalize(centers[c].unsqueeze(0), dim=1)
                dist = 1.0 - (p @ cen.t()).squeeze(1)  # cosine distance
            else:
                dist = torch.norm(pts - centers[c].unsqueeze(0), dim=1)
            q_idxs.append(ids[torch.argmin(dist)])

        q_idxs = torch.stack(q_idxs)
        
        self._labels = labels.detach()
        self._centers = centers.detach()
        self._center_assign = q_idxs.detach()
        
        return q_idxs

    def query_incluster_n(self, n):
        """
        n개 클러스터로 kmeans → 각 클러스터 중심에서
        해당 클러스터에 속한 샘플 중 가장 가까운 인덱스를 반환.
        """
        embeddings = self.img_feats
        index = torch.arange(len(embeddings), device=embeddings.device)

        # k-means 학습
        kmeans = KMeans(n_clusters=n, mode=self.mode, verbose=False)        # euclidean으로 되어 있었음 (v11까지)
        labels = kmeans.fit_predict(embeddings)
        centers = kmeans.centroids  # [n, D]

        q_idxs = []
        chosen = torch.zeros(len(embeddings), dtype=torch.bool, device=embeddings.device)

        for c in range(n):
            mask = (labels == c)
            cen = centers[c].unsqueeze(0)

            if mask.any():
                pts = embeddings[mask]
                ids = index[mask]
                if self.mode == "cosine":
                    p = F.normalize(pts, dim=1)
                    cen_n = F.normalize(cen, dim=1)
                    dist = 1.0 - (p @ cen_n.t()).squeeze(1)
                else:
                    dist = torch.norm(pts - cen, dim=1)
                # 클러스터 내부에서만 선택
                pick = ids[torch.argmin(dist)]
            else:
                # 빈 클러스터: 전체에서 선택(이미 선택된 건 제외)
                if self.mode == "cosine":
                    p_all = F.normalize(embeddings, dim=1)
                    cen_n = F.normalize(cen, dim=1)
                    dist_all = 1.0 - (p_all @ cen_n.t()).squeeze(1)
                else:
                    dist_all = torch.norm(embeddings - cen, dim=1)

                dist_all = dist_all.masked_fill(chosen, float('inf'))  # 중복 방지
                pick = index[torch.argmin(dist_all)]

            q_idxs.append(pick)
            chosen[pick] = True

        q_idxs = torch.stack(q_idxs)
        
        self._labels = labels.detach()
        self._centers = centers.detach()
        self._center_assign = q_idxs.detach()
        
        return q_idxs
    
    @property
    def labels(self):   return self._labels
    @property
    def centers(self):  return self._centers
    @property
    def chosen(self):   return self._center_assign

    def visualize(self, selected_indices: torch.Tensor, filename='kmeans_vis.png',title=None, max_points=None):
        """
        - 각 클러스터 색상으로 전체 포인트 표시
        - 선택된 인덱스(*)와 센트로이드(X) 표시
        - feats 샘플링 시에도 selected 매칭 정확히 수행
        """
        assert self._labels is not None and self._centers is not None, \
            "Run query(n) first so that labels/centers are set."

        feats = self.img_feats
        labels = self._labels
        centers = self._centers

        # -------- 샘플링(선택) --------
        N = feats.size(0)
        if (max_points is not None) and (N > max_points):
            with torch.no_grad():
                idx_chunks = []
                k = int(labels.max().item()) + 1
                per_k = max(1, max_points // k)
                for c in range(k):
                    cand = torch.nonzero(labels == c, as_tuple=False).flatten()
                    if cand.numel() > per_k:
                        cand = cand[torch.randperm(cand.numel(), device=cand.device)[:per_k]]
                    idx_chunks.append(cand)
                idx = torch.cat(idx_chunks, dim=0)
        else:
            idx = torch.arange(N, device=feats.device)

        feats_plot = feats[idx]
        labels_plot = labels[idx]

        # -------- 같은 PCA basis로 투영 --------
        with torch.no_grad():
            Z, V, mean = pca_2d_fit(feats_plot.float())  # feats_plot 기준으로 basis 학습
            Zc = (centers.float() - mean) @ V[:, :2]     # 같은 basis에 centroid 투영

        z_np  = Z.detach().cpu().numpy()
        zc_np = Zc.detach().cpu().numpy()
        labs_np = labels_plot.detach().cpu().numpy()

        # -------- 선택 인덱스 매칭(샘플링 고려) --------
        sel = selected_indices.detach().to(idx.device)
        if idx.numel() == N and torch.all(idx == torch.arange(N, device=idx.device)):
            # 샘플링 안함 → 바로 마스크 가능
            sel_mask = torch.zeros(N, dtype=torch.bool, device=idx.device)
            sel_mask[sel] = True
            z_sel_np = Z[sel_mask[idx]].detach().cpu().numpy()
        else:
            # 샘플링함 → idx 안에 포함된 selected만 표시
            sel_mask_plot = torch.isin(idx, sel)
            z_sel_np = Z[sel_mask_plot].detach().cpu().numpy()

        # -------- 그리기 --------
        import matplotlib.pyplot as plt
        import distinctipy
        colors = distinctipy.get_colors(self._centers.size(0))
        
        plt.figure(figsize=(9, 8))
        plt.scatter(z_np[:,0], z_np[:,1], c=[colors[i] for i in labs_np],
            s=6, alpha=0.6)

        if z_sel_np.size > 0:
            plt.scatter(z_sel_np[:, 0], z_sel_np[:, 1], marker='*', s=160, 
                        edgecolor='k', facecolor='White', linewidths=1.2, label='Selected')

        plt.scatter(zc_np[:, 0], zc_np[:, 1], marker='X', s=200, 
                    edgecolor='k', facecolor='White', linewidths=1.5, label='Centroid')

        plt.legend(loc='best', frameon=True)
        plt.title(title or f'K-means clustering (k={centers.size(0)})')
        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close()
        print(f"[KMeansCluster] Saved visualization to: {filename}")

class OnlineCentroidBank:
    """
    - init_centers: [K,D] 초기 중심 (kmeans 결과)
    - update(real_feats): 실배치의 이미지를 현재 중심에 할당 후 EMA로 중심 갱신
    - pull_loss(syn_feats, syn_ids): 합성 이미지 feature를 대응 중심으로 끌어당기는 손실 계산
    """
    def __init__(self, init_centers: torch.Tensor, metric: str = "cosine", momentum: float = 0.05):
        self.metric = metric  # 'cosine' | 'euclidean'
        self.m = momentum
        if self.metric == "cosine":
            init_centers = F.normalize(init_centers, dim=1)
        self.centers = init_centers.detach().clone()  # [K,D]

    @torch.no_grad()
    def _assign(self, feats: torch.Tensor) -> torch.Tensor:
        """feats → 가장 가까운 중심 id (고차원에서)"""
        if self.metric == "cosine":
            feats_n = F.normalize(feats, dim=1)
            sims = feats_n @ self.centers.t()        # [B,K]
            return torch.argmax(sims, dim=1)
        else:
            d = torch.cdist(feats, self.centers)     # [B,K]
            return torch.argmin(d, dim=1)

    @torch.no_grad()
    def update(self, real_feats: torch.Tensor):
        """
        실배치(real image features)를 현재 중심에 할당 후
        각 클러스터의 배치 평균으로 EMA 갱신.
        """
        if real_feats.numel() == 0:
            return
        ids = self._assign(real_feats)  # [B]
        for c in torch.unique(ids):
            c = int(c.item())
            pts = real_feats[ids == c]
            mu = pts.mean(dim=0, keepdim=True)             # [1,D]
            if self.metric == "cosine":
                mu = F.normalize(mu, dim=1)
            self.centers[c:c+1] = F.normalize(
                (1.0 - self.m) * self.centers[c:c+1] + self.m * mu, dim=1
            ) if self.metric == "cosine" else \
            (1.0 - self.m) * self.centers[c:c+1] + self.m * mu

    def pull_loss(self, syn_feats: torch.Tensor, syn_ids: torch.Tensor) -> torch.Tensor:
        """
        syn_feats: [B,D] 합성 이미지 feature (정규화 권장)
        syn_ids:   [B] 각 합성 샘플이 따라갈 중심 id (여기서는 KMeans 생성 순서 그대로 idx 사용)
        """
        C = self.centers[syn_ids]  # [B,D]
        if self.metric == "cosine":
            syn = F.normalize(syn_feats, dim=1)
            C   = F.normalize(C, dim=1)
            return (1.0 - (syn * C).sum(dim=1)).mean()
        else:
            return torch.mean(torch.sum((syn_feats - C) ** 2, dim=1))