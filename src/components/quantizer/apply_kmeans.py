import joblib
import torch
import torch.nn as nn


class BatchKMeans(nn.Module):
    def __init__(self, km_path: str = None, device: str = "cuda"):
        """
        km_path: path to your joblib-saved sklearn.KMeans
        device: "cuda" or "cpu"
        """
        super().__init__()
        # load cluster centers: shape (n_clusters, D)
        if km_path is not None:
            km = joblib.load(km_path)
            centers = torch.from_numpy(km.cluster_centers_).float()
        else:
            centers = torch.randn(256, 512).float()  # default random centers
        
        # register as buffers so .to(device) moves them
        self.register_buffer("centers", centers)                 
        self.register_buffer("centers_norm", (centers**2).sum(1))# (n_clusters,)
        
        self.to(device)

    def forward(self, x: torch.Tensor, return_indices: bool = False):
        """
        x: Tensor of shape (B, D, T)
        return_indices: if True  → LongTensor of shape (B, T) with cluster IDs
                        else       → FloatTensor of shape (B, D, T) with centroids
        """
        B, D, T = x.shape
        
        # flatten time & batch dims into one
        flat = x.transpose(1, 2).reshape(-1, D)  # (B*T, D)
        
        # squared‐Euclidean distance to each center:
        dist2 = (
            flat.pow(2).sum(1, keepdim=True)           # (B*T, 1)
            - 2 * flat @ self.centers.t()              # (B*T, n_clusters)
            + self.centers_norm.unsqueeze(0)           # (1,  n_clusters)
        )
        
        # nearest cluster for each vector
        ids = dist2.argmin(dim=1)                     # (B*T,)
        
        if return_indices:
            return ids.view(B, T)                     # (B, T)
        else:
            # gather centroids and restore (B, D, T)
            quant = self.centers[ids]                 # (B*T, D)
            quant = quant.view(B, T, D)               # (B, T, D)
            return quant.transpose(1, 2)              # (B, D, T)