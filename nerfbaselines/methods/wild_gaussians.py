# We extend WildGaussians to add demo export
import torch
from wildgaussians.method import WildGaussians as _WildGaussians  # type: ignore


class WildGaussians(_WildGaussians):
    @torch.no_grad()
    def export_demo(self, path: str, *, options=None):
        options = options or {}
        from ._gaussian_splatting_demo import export_demo

        gaussians = self.model.get_gaussians()
        device = gaussians["xyz"].device
        features = gaussians["features"].clamp_max(1.0)
        shdim = (self.config.sh_degree + 1) ** 2
        if self.config.appearance_enabled:
            embedding_np = options.get("embedding", None)
            if embedding_np is None:
                embedding = self.model.get_embedding(None)
            else:
                embedding = torch.from_numpy(embedding_np).to(device)
            embedding_expanded = embedding[None].repeat(len(gaussians["xyz"]), 1)
            colors_toned = self.model.appearance_mlp(self.model.embeddings, embedding_expanded, features).clamp_max(1.0)
            spherical_harmonics = colors_toned.view(-1, shdim, 3).transpose(1, 2).contiguous().clamp_max(1.0)
        elif features.shape[-1] == 3:
            C0 = 0.28209479177387814
            colors = features[..., None]
            spherical_harmonics = (colors - 0.5) / C0
        else:
            assert features.shape[-1] == shdim * 3
            shdim = (self.config.sh_degree + 1) ** 2
            spherical_harmonics = features.view(-1, shdim, 3).transpose(1, 2).contiguous()

        export_demo(path, 
                    options=options,
                    xyz=gaussians["xyz"].detach().cpu().numpy(),
                    scales=gaussians["scales"].detach().cpu().numpy(),
                    opacities=gaussians["opacities"].detach().cpu().numpy(),
                    quaternions=gaussians["rotations"].detach().cpu().numpy(),
                    spherical_harmonics=spherical_harmonics.detach().cpu().numpy())
