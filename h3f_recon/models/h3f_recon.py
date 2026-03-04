import torch
import torch.nn as nn
import torch.nn.functional as F

from h3f_recon.config import DataConfig, ModelConfig
from h3f_recon.models.decoder import LocalFusionDecoder
from h3f_recon.models.point_geometry import PointGeometryEncoder
from h3f_recon.models.positional_encoding import FourierPositionalEncoding
from h3f_recon.models.sparse_hierarchy import SparseFeatureHierarchy


class H3FRecon(nn.Module):
    def __init__(self, model_cfg: ModelConfig, data_cfg: DataConfig) -> None:
        super().__init__()
        self.top_k = model_cfg.top_k
        self.use_point_geo = bool(model_cfg.use_point_geo)

        self.sparse_hierarchy = SparseFeatureHierarchy(
            resolutions=model_cfg.resolutions,
            cells_per_level=model_cfg.cells_per_level,
            feature_dims=model_cfg.feature_dims,
            world_bound=data_cfg.world_bound,
            max_candidates=model_cfg.max_candidates,
        )
        self.positional_encoding = FourierPositionalEncoding(model_cfg.num_frequencies)

        point_geo_dim = int(model_cfg.point_geo_dim) if self.use_point_geo else 0
        if self.use_point_geo:
            self.point_geo_encoder = PointGeometryEncoder(
                k_neighbors=model_cfg.point_geo_k,
                out_dim=model_cfg.point_geo_dim,
                hidden_dim=model_cfg.point_geo_hidden_dim,
                max_context_points=model_cfg.point_geo_max_context,
            )
        else:
            self.point_geo_encoder = None

        decoder_in_dim = self.positional_encoding.out_dim + sum(model_cfg.feature_dims) + point_geo_dim
        self.local_decoder = LocalFusionDecoder(
            in_dim=decoder_in_dim,
            hidden_dim=model_cfg.hidden_dim,
            num_layers=model_cfg.decoder_layers,
        )

    def forward(
        self,
        points: torch.Tensor,
        context_points: torch.Tensor | None = None,
        context_normals: torch.Tensor | None = None,
        return_intermediates: bool = False,
    ) -> dict[str, torch.Tensor]:
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError("points must have shape [N, 3]")

        sparse_query = self.sparse_hierarchy(points, top_k=self.top_k)
        encoded_local = self.positional_encoding(sparse_query["local_coords"])
        decoder_parts = [encoded_local, sparse_query["features"]]

        if self.use_point_geo and self.point_geo_encoder is not None:
            geo_feat = self.point_geo_encoder(
                query_points=points,
                context_points=context_points,
                context_normals=context_normals,
            )
            geo_feat = geo_feat.unsqueeze(1).expand(-1, sparse_query["features"].shape[1], -1)
            decoder_parts.append(geo_feat)

        decoder_input = torch.cat(decoder_parts, dim=-1)

        local_sdf, blend_logits, local_uncertainty = self.local_decoder(decoder_input)
        blend_logits = blend_logits + torch.log(sparse_query["quality"] + 1e-6)

        weights = torch.softmax(blend_logits, dim=1)
        fused_sdf = (weights * local_sdf).sum(dim=1)
        fused_uncertainty = (weights * F.softplus(local_uncertainty)).sum(dim=1)

        out = {
            "sdf": fused_sdf,
            "uncertainty": fused_uncertainty,
        }

        if return_intermediates:
            out.update(
                {
                    "local_sdf": local_sdf,
                    "weights": weights,
                    "blend_logits": blend_logits,
                    "local_coords": sparse_query["local_coords"],
                    "candidate_quality": sparse_query["quality"],
                    "point_geo": geo_feat if self.use_point_geo and self.point_geo_encoder is not None else None,
                }
            )

        return out
