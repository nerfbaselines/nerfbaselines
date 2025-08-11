"""
We patched 3DGRUT with support for OPENCV and OPENCV_FULL camera models.
"""
from ._patching import Context

import_context = Context()
import_context.apply_patch(r"""
diff --git a/threedgrut/datasets/dataset_colmap.py b/threedgrut/datasets/dataset_colmap.py
index 89b2a45..3244c39 100644
--- a/threedgrut/datasets/dataset_colmap.py
+++ b/threedgrut/datasets/dataset_colmap.py
@@ -24,2 +24,3 @@ import torch
 from torch.utils.data import Dataset
+import nerfbaselines.cameras as nb_cameras
 
@@ -132,3 +133,3 @@ class ColmapDataset(Dataset, BoundedMultiViewDataset, DatasetVisualization):
 
-        def create_pinhole_camera(focalx, focaly, w, h):
+        def create_pinhole_camera(focalx, focaly, cx, cy, w, h, d=None):
             # Generate UV coordinates
@@ -137,2 +138,11 @@ class ColmapDataset(Dataset, BoundedMultiViewDataset, DatasetVisualization):
             out_shape = (1, h, w, 3)
+            camera_model = "pinhole"
+            if d is None:
+                d = [0] * 8
+            else:
+                if len(d) <= 4:
+                    camera_model = "opencv"
+                else:
+                    camera_model = "full_opencv"
+                d = d.tolist() + [0] * 8
             params = OpenCVPinholeCameraModelParameters(
@@ -142,9 +152,21 @@ class ColmapDataset(Dataset, BoundedMultiViewDataset, DatasetVisualization):
                 focal_length=np.array([focalx, focaly], dtype=np.float32),
-                radial_coeffs=np.zeros((6,), dtype=np.float32),
-                tangential_coeffs=np.zeros((2,), dtype=np.float32),
+                radial_coeffs=np.array((d[0], d[1], d[4], d[5], d[6], d[7]), dtype=np.float32),
+                tangential_coeffs=np.array((d[2], d[3]), dtype=np.float32),
                 thin_prism_coeffs=np.zeros((4,), dtype=np.float32),
             )
-            rays_o_cam, rays_d_cam = pinhole_camera_rays(
-                u, v, focalx, focaly, w, h, self.ray_jitter
-            )
+            xs, ys = u, v
+            if self.ray_jitter is not None:
+                jitter = self.ray_jitter(u.shape).numpy()
+                xs = xs + jitter[:, 0]
+                ys = ys + jitter[:, 1]
+            else:
+                xs = xs + 0.5
+                ys = ys + 0.5
+            xs = (xs - cx) / focalx
+            ys = (ys - cy) / focaly
+            uv = np.stack((xs, ys), axis=-1)
+            uv = nb_cameras._undistort(nb_cameras.camera_model_to_int(camera_model), np.array(d)[None], uv, xnp=np)
+            rays_d_cam = np.concatenate((uv, np.ones_like(uv[..., :1])), -1)
+            rays_d_cam = rays_d_cam / np.linalg.norm(rays_d_cam, axis=-1, keepdims=True)
+            rays_o_cam = np.zeros_like(rays_d_cam)
             return (
@@ -228,4 +250,6 @@ class ColmapDataset(Dataset, BoundedMultiViewDataset, DatasetVisualization):
                 focal_length = intr.params[0] / scaling_factor
+                cx = intr.params[1] / scaling_factor
+                cy = intr.params[2] / scaling_factor
                 self.intrinsics[intr.id] = create_pinhole_camera(
-                    focal_length, focal_length, width, height
+                    focal_length, focal_length, cx, cy, width, height
                 )
@@ -235,4 +259,16 @@ class ColmapDataset(Dataset, BoundedMultiViewDataset, DatasetVisualization):
                 focal_length_y = intr.params[1] / scaling_factor
+                cx = intr.params[2] / scaling_factor
+                cy = intr.params[3] / scaling_factor
+                self.intrinsics[intr.id] = create_pinhole_camera(
+                    focal_length_x, focal_length_y, cx, cy, width, height
+                )
+
+            elif intr.model == "OPENCV" or intr.model == "FULL_OPENCV":
+                focal_length_x = intr.params[0] / scaling_factor
+                focal_length_y = intr.params[1] / scaling_factor
+                cx = intr.params[2] / scaling_factor
+                cy = intr.params[3] / scaling_factor
+                d = intr.params[4:]
                 self.intrinsics[intr.id] = create_pinhole_camera(
-                    focal_length_x, focal_length_y, width, height
+                    focal_length_x, focal_length_y, cx, cy, width, height, d
                 )
diff --git a/threedgrut/model/background.py b/threedgrut/model/background.py
index 9dec3b4..5d6c671 100644
--- a/threedgrut/model/background.py
+++ b/threedgrut/model/background.py
@@ -65,8 +65,2 @@ class BackgroundColor(BaseBackground):
 
-        assert self.background_color_type in [
-            "white",
-            "black",
-            "random",
-        ], "Background color must be one of 'white', 'black', 'random'"
-
         if self.background_color_type == "white":
@@ -78,2 +72,4 @@ class BackgroundColor(BaseBackground):
             self.color = torch.zeros((3,), dtype=torch.float32, device=self.device)
+        else:
+            self.color = torch.tensor(self.background_color_type, dtype=torch.float32, device=self.device)
 
"""[1:-1])
