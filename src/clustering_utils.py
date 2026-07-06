import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gc

from fast_pytorch_kmeans import KMeans

def l2_normalize(x, dim=-1, eps=1e-12):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def pca_2d_fit(x: torch.Tensor):
    mean = x.mean(dim=0, keepdim=True)
    x0 = x - mean
    U, S, V = torch.pca_lowrank(x0, q=2)
    Z = x0 @ V[:, :2]  # [N,2]
    return Z, V, mean 

class KMeansCluster:
    def __init__(self, img_feats: torch.Tensor, device=None, mode: str = "cosine"):
        
        self.img_feats = img_feats
        self.device = device or (img_feats.device if isinstance(img_feats, torch.Tensor) else 'cpu')
        self._labels = None          # [N]
        self._centers = None         # [k, D]
        self._center_assign = None
        self.mode = mode
    
    def free(self):
        # Drop PyTorch CUDA tensors
        for name, val in list(self.__dict__.items()):
            if torch.is_tensor(val) and val.is_cuda:
                setattr(self, name, None)
        self.img_feats = None
        self._labels = None          # [N]
        self._centers = None         # [k, D]
        self._center_assign = None
        gc.collect()
        torch.cuda.empty_cache()
    
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
        embeddings = self.img_feats
        index = torch.arange(len(embeddings), device=embeddings.device)

        kmeans = KMeans(n_clusters=n, mode=self.mode, verbose=False)
        labels = kmeans.fit_predict(embeddings)
        centers = kmeans.centroids  # [n, D]

        dist_matrix = self.euclidean_dist(centers, embeddings)  # [n, N]
        q_idxs = index[torch.argmin(dist_matrix, dim=1)]        # [n]

        self._labels = labels.detach()
        self._centers = centers.detach()
        self._center_assign = q_idxs.detach()

        return q_idxs

    def query_incluster(self, n):
        embeddings = self.img_feats
        index = torch.arange(len(embeddings), device=embeddings.device)

        kmeans = KMeans(n_clusters=n, mode=self.mode, verbose=False)
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
        embeddings = self.img_feats
        index = torch.arange(len(embeddings), device=embeddings.device)

        # k-means 학습
        kmeans = KMeans(n_clusters=n, mode=self.mode, verbose=False)
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

                dist_all = dist_all.masked_fill(chosen, float('inf'))
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
        
        assert self._labels is not None and self._centers is not None, \
            "Run query(n) first so that labels/centers are set."

        feats = self.img_feats
        labels = self._labels
        centers = self._centers

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

        with torch.no_grad():
            Z, V, mean = pca_2d_fit(feats_plot.float())
            Zc = (centers.float() - mean) @ V[:, :2]   

        z_np  = Z.detach().cpu().numpy()
        zc_np = Zc.detach().cpu().numpy()
        labs_np = labels_plot.detach().cpu().numpy()

        sel = selected_indices.detach().to(idx.device)
        if idx.numel() == N and torch.all(idx == torch.arange(N, device=idx.device)):
            sel_mask = torch.zeros(N, dtype=torch.bool, device=idx.device)
            sel_mask[sel] = True
            z_sel_np = Z[sel_mask[idx]].detach().cpu().numpy()
        else:
            sel_mask_plot = torch.isin(idx, sel)
            z_sel_np = Z[sel_mask_plot].detach().cpu().numpy()

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
    
    def __init__(self, init_centers: torch.Tensor, metric: str = "cosine", momentum: float = 0.05):
        self.metric = metric  # 'cosine' | 'euclidean'
        self.m = momentum
        if self.metric == "cosine":
            init_centers = F.normalize(init_centers, dim=1)
        self.centers = init_centers.detach().clone()  # [K,D]

    @torch.no_grad()
    def _assign(self, feats: torch.Tensor) -> torch.Tensor:
        if self.metric == "cosine":
            feats_n = F.normalize(feats, dim=1)
            sims = feats_n @ self.centers.t()        # [B,K]
            return torch.argmax(sims, dim=1)
        else:
            d = torch.cdist(feats, self.centers)     # [B,K]
            return torch.argmin(d, dim=1)

    @torch.no_grad()
    def update(self, real_feats: torch.Tensor):
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
        C = self.centers[syn_ids]  # [B,D]
        if self.metric == "cosine":
            syn = F.normalize(syn_feats, dim=1)
            C   = F.normalize(C, dim=1)
            return (1.0 - (syn * C).sum(dim=1)).mean()
        else:
            return torch.mean(torch.sum((syn_feats - C) ** 2, dim=1))
