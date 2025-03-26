import logging
import shutil
import subprocess
import numpy as np
import json
from typing import Optional
from collections import defaultdict
import os
import tempfile
from nerfbaselines import Method, Dataset, Cameras, RenderOptions, camera_model_from_int, ModelInfo, MethodInfo
from nerfbaselines.datasets import _colmap_utils as colmap_utils


logger = logging.getLogger(__name__)


def _pad_poses(poses):
    if poses.shape[-2] == 3:
        lastrow = np.zeros_like(poses[..., 0:1, :])
        lastrow[..., 0, 3] = 1.0
        poses = np.concatenate([poses, lastrow], axis=-2)
    return poses


def _get_colmap_sfm_reconstruction(dataset: Dataset, image_ids=None):
    colmap_images = {}
    colmap_cameras = {}
    colmap_points3D = {}
    cameras = dataset["cameras"]

    def _notnonelist(x):
        if x is None:
            return []
        return x

    image_ids_map = defaultdict(list)
    for i, point3D_idxs in enumerate(_notnonelist(dataset.get("images_points3D_indices"))):
        for ptidx in point3D_idxs:
            image_ids_map[ptidx].append(i+1)

    points3D_rgb = dataset["points3D_rgb"]
    for i, xyz in enumerate(_notnonelist(dataset.get("points3D_xyz"))):
        assert points3D_rgb is not None, "RGB values are required for 3D points"
        rgb = points3D_rgb[i]
        error = 0
        _image_ids = np.array(image_ids_map[i], dtype=np.int64)
        point2D_idxs = np.zeros(len(_image_ids), dtype=np.int64)
        colmap_points3D[i+1] = colmap_utils.Point3D(i+1, xyz, rgb, error, _image_ids, point2D_idxs)

    for i, cam in enumerate(cameras):
        width, height = cam.image_sizes
        cam_model = camera_model_from_int(int(cam.camera_models))
        fx, fy, cx, cy = cam.intrinsics
        if cam_model == "pinhole":
            model = "PINHOLE"
            params = [fx, fy, cx, cy]
        elif cam_model == "opencv":
            model = "OPENCV"
            k1, k2, p1, p2, k3, k4, *_ = cam.distortion_parameters
            params = [fx, fy, cx, cy, k1, k2, p1, p2]
        elif cam_model == "opencv_fisheye":
            model = "OPENCV_FISHEYE"
            k1, k2, _, _, k3, k4, *_ = cam.distortion_parameters
            params = [fx, fy, cx, cy, k1, k2, k3, k4]
        elif cam_model == "full_opencv":
            model = "FULL_OPENCV"
            k1, k2, p1, p2, k3, k4, k5, k6 = cam.distortion_parameters
            params = [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6]
        else:
            raise ValueError(f"Unknown camera model {cam_model}")

        colmap_cameras[i+1] = colmap_utils.Camera(i+1, model, width, height, params)

        name = os.path.relpath(dataset["image_paths"][i], dataset["image_paths_root"])
        image_id = i+1
        if image_ids is not None:
            image_id = image_ids[name]
        R = cam.poses[:3, :3].T
        qvec = colmap_utils.rotmat2qvec(R)
        tvec = -np.matmul(R, cam.poses[:3, 3]).reshape(3)
        point3D_ids = np.array([], dtype=np.int64)
        xys = np.zeros((0, 2), dtype=np.float32)
        images_points3D_indices = dataset.get("images_points3D_indices")
        if images_points3D_indices is not None:
            point3D_ids = images_points3D_indices[i] + 1
            xys = np.zeros((len(point3D_ids), 2), dtype=np.float32)
        colmap_images[image_id] = colmap_utils.Image(image_id, qvec, tvec, i+1, name, xys, point3D_ids)
    return colmap_cameras, colmap_images, colmap_points3D


def _write_colmap_dataset(dataset: Dataset, path: str, image_ids=None):
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "sparse"), exist_ok=True)
    cameras, images, points3D = _get_colmap_sfm_reconstruction(dataset, image_ids=image_ids)
    colmap_utils.write_cameras_binary(cameras, os.path.join(path, "sparse", "cameras.bin"))
    colmap_utils.write_images_binary(images, os.path.join(path, "sparse", "images.bin"))
    colmap_utils.write_points3D_binary(points3D, os.path.join(path, "sparse", "points3D.bin"))


def _fill_missing_colmap_points3D(dataset: Dataset, hparams):
    # In case of no 3D points, we get the sparse model first
    # to use it for patch_match_stereo to select closest-looking views
    del hparams

    import sqlite3
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1) Write the list of images
        with open(os.path.join(tmpdir, "images.txt"), "w") as f:
            for img in dataset["image_paths"]:
                f.write(f"{os.path.relpath(img, dataset['image_paths_root'])}\n")

        # 1) Run feature extraction
        subprocess.check_call([
            "colmap", "feature_extractor",
            "--database_path", "database.db",
            "--image_list_path", "images.txt",
            "--image_path", dataset["image_paths_root"]], cwd=tmpdir)
        logger.info("Features extracted")

        # 2) Run feature matching
        subprocess.check_call([
            "colmap", "exhaustive_matcher",
            "--database_path", "database.db"], cwd=tmpdir)

        # 3) Write the sparse reconstruction with correct image ids
        connection = sqlite3.Connection(os.path.join(tmpdir, "database.db"))
        try:
            image_ids = dict(connection.execute("SELECT name, image_id FROM images"))
        finally:
            connection.close()
        _write_colmap_dataset(dataset, tmpdir, image_ids=image_ids)

        # 4) Triangulate the points
        os.makedirs(os.path.join(tmpdir, "triangulated"), exist_ok=True)
        subprocess.check_call([
            "colmap", "point_triangulator",
            "--image_path", dataset["image_paths_root"],
            "--input_path", os.path.join(tmpdir, "sparse"),
            "--output_path", os.path.join(tmpdir, "triangulated"),
            "--database_path", "database.db"], cwd=tmpdir)

        # 5) Read the points and update the dataset
        points3D = colmap_utils.read_points3D_binary(os.path.join(tmpdir, "triangulated", "points3D.bin"))
        images = colmap_utils.read_images_binary(os.path.join(tmpdir, "triangulated", "images.bin"))
        dataset["points3D_xyz"] = np.array([x.xyz for x in points3D.values()])
        dataset["points3D_rgb"] = np.array([x.rgb for x in points3D.values()])
        inverse_index = {x.id: i for i, x in enumerate(points3D.values())}
        dataset["images_points3D_indices"] = [np.zeros((0,), dtype=np.int64) for _ in range(len(images))]
        for i, full_path in enumerate(dataset["image_paths"]):
            relpath = os.path.relpath(full_path, dataset["image_paths_root"])
            image_id = image_ids[relpath]
            dataset["images_points3D_indices"][i] = \
                np.array([inverse_index[x] for x in images[image_id].point3D_ids if x != -1], dtype=np.int64)


def _create_raymond_lights():
    from pyrender import DirectionalLight  # type: ignore
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3,:3] = np.c_[x,y,z]
        nodes.append((
            DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix
        ))
    return nodes


class ColmapMVS(Method):
    def __init__(self, *,
                 checkpoint: Optional[str] = None,
                 train_dataset: Optional[Dataset] = None,
                 config_overrides: Optional[dict] = None):
        self._checkpoint = checkpoint

        # Default hparams
        hparams = {
            "mesher": "poisson",
            "PatchMatchStereo.geom_consistency": "true",
        }
        if checkpoint is not None:
            if os.path.exists(os.path.join(checkpoint, "hparams.json")):
                with open(os.path.join(checkpoint, "hparams.json"), "r") as f:
                    hparams.update(json.load(f))
                hparams.update(config_overrides or {})
                assert config_overrides is None, "config_overrides must be None if checkpoint is provided"
            else:
                hparams.update(config_overrides or {})
        else:
            hparams.update(config_overrides or {})

        self._hparams = hparams
        self._train_dataset = train_dataset
        
        # For newly trained model, we store it in 
        # a temporary directory before save is called
        self._tmpdir = None
        if self._checkpoint is None:
            self._tmpdir = tempfile.TemporaryDirectory()
            self._model_path = self._tmpdir.name
        else:
            self._model_path = self._checkpoint

        # Setup renderer
        try:
            os.environ["PYOPENGL_PLATFORM"] = "egl"
            import pyrender  # type: ignore
            from pyrender import Renderer as PyRenderer  # type: ignore
            from pyrender.platforms import egl  # type: ignore
        except ImportError as e:
            if "Unable to load EGL library" in str(e):
                raise RuntimeError("No suitable GPU found for rendering") from e
            raise

        camera = pyrender.IntrinsicsCamera(fx=200., fy=200.0, cx=100.0, cy=100.0, znear=0.001, zfar=10_000.0)
        self._scene = pyrender.Scene(
            bg_color=np.array([0., 0., 0., 0.], dtype=np.float32)
        )
        self._model = None
        self._camera = self._scene.add(camera, pose=np.eye(4))
        self._lights = []
        for light, pose in _create_raymond_lights():
            self._lights.append(
                self._scene.add(light, pose=pose, parent_node=self._camera)
            )

        device_id = int(os.environ.get('EGL_DEVICE_ID', '0'))
        egl_device = egl.get_device_by_index(device_id)
        self._platform = egl.EGLPlatform(200, 200, device=egl_device)
        self._platform.init_context()
        self._platform.make_current()
        self._renderer = PyRenderer(200, 200)
        self._renderer.point_size = 1.0
        if self._checkpoint is not None:
            self._load_mesh()

    def _load_mesh(self):
        import trimesh  # type: ignore
        import pyrender  # type: ignore
        mesh_path = os.path.join(self._model_path, "mesh.ply")
        mesh = trimesh.load_mesh(mesh_path)
        if isinstance(mesh, trimesh.Scene) and mesh.is_empty:
            raise RuntimeError("Empty mesh")
        if self._model is not None:
            for node in self._model:
                self._scene.remove_node(node)
        if not isinstance(mesh, list):
            mesh = [mesh]
        self._model = [
            self._scene.add(pyrender.Mesh.from_trimesh(m))
            for m in mesh
        ]

    @classmethod
    def get_method_info(cls):
        return MethodInfo(
            method_id="",  # Will be filled in by the registry
            required_features=frozenset(("color", "points3D_xyz", "points3D_rgb", "images_points3D_indices")),
            supported_camera_models=frozenset(("pinhole", "opencv", "opencv_fisheye", "full_opencv")),
            supported_outputs=("color", "depth"),
            viewer_default_resolution=512,
        )

    def get_info(self):
        return ModelInfo(
            **self.get_method_info(),
            num_iterations=1,
            loaded_checkpoint=self._checkpoint,
            loaded_step=(1 if self._checkpoint is not None else None),
            hparams=self._hparams,
        )

    def train_iteration(self, step):
        assert self._train_dataset is not None, "No training dataset provided"
        assert step == 0, "COLMAP is not an iterative method"

        # Write the sparse reconstruction
        with tempfile.TemporaryDirectory() as tmpdir:
            logger.info("Temporary directory:" + tmpdir)

            # 0) Fill missing points
            if self._train_dataset.get("images_points3D_indices") is None:
                # NOTE: 2D-3D correspondences are required for finding closest matches for stereo
                logger.warning("No 3D-2D correspondences found, filling missing points")
                _fill_missing_colmap_points3D(self._train_dataset, self._hparams)

            _write_colmap_dataset(self._train_dataset, tmpdir)
            logger.info("Sparse reconstruction written")

            # 1) Run undistort
            image_path = self._train_dataset["image_paths_root"]
            subprocess.check_call([
                "colmap", "image_undistorter",
                "--image_path", image_path,
                "--input_path", "sparse",
                "--output_path", "dense",
                "--output_type=COLMAP",
                "--max_image_size=2000"], cwd=tmpdir)
            logger.info("Undistorted images saved to dense")

            # 2) Run stereo
            subprocess.check_call([
                "colmap", "patch_match_stereo",
                "--workspace_path", "dense",
                "--workspace_format", "COLMAP",
                *(f"--{x}={y}" for x, y in self._hparams.items() if x.startswith("PatchMatchStereo."))
            ], cwd=tmpdir)
            logger.info("Stereo reconstruction done")

            # 3) Run fusion
            subprocess.check_call([
                "colmap", "stereo_fusion",
                "--workspace_path", "dense",
                *(f"--{x}={y}" for x, y in self._hparams.items() if x.startswith("StereoFusion.")),
                "--output_path", "dense/fused.ply"], cwd=tmpdir)
            logger.info(f"Fused model saved to {tmpdir}/dense/fused.ply")

            # 4) Run mesher
            mesher = self._hparams.get("mesher", "poisson")
            mesh_path = os.path.join(self._model_path, "mesh.ply")
            if mesher == "poisson":
                logging.info("Running poisson mesher")
                subprocess.check_call([
                    "colmap", "poisson_mesher",
                    "--input_path", "dense/fused.ply",
                    *(f"--{x}={y}" for x, y in self._hparams.items() if x.startswith("PoissonMeshing.")),
                    "--output_path", mesh_path], cwd=tmpdir)
            elif mesher == "delaunay":
                logging.info("Running delaunay mesher")
                subprocess.check_call([
                    "colmap", "delaunay_mesher",
                    "--input_path", "dense",
                    *(f"--{x}={y}" for x, y in self._hparams.items() if x.startswith("DelaunayMeshing.")),
                    "--output_path", mesh_path], cwd=tmpdir)
            else:
                raise ValueError(f"Unknown mesher {mesher}, must be 'poisson' or 'delaunay'")
            logging.info("Mesh saved to " + mesh_path)

            # Load the mesh here
            self._load_mesh()
            logging.info("Mesh loaded")
            return {}

    def save(self, path):
        os.makedirs(path, exist_ok=True)

        # Save parameters
        with open(os.path.join(path, "hparams.json"), "w") as f:
            json.dump(self._hparams, f)

        # Save model
        if path != self._model_path:
            for f in os.listdir(self._model_path):
                shutil.copy(os.path.join(self._model_path, f), os.path.join(path, f))

    def render(self, camera: Cameras, *, options: Optional[RenderOptions] = None):
        from pyrender import RenderFlags  # type: ignore
        del options
        assert self._platform is not None, "Method is already destroyed"
        assert self._renderer is not None, "Method is already destroyed"
        flags = RenderFlags.OFFSCREEN | RenderFlags.FLAT | RenderFlags.RGBA
        assert self._platform.supports_framebuffers(), "Platform does not support framebuffers"
        self._platform.make_current()
        cam = camera.item()
        try:
            pose = _pad_poses(cam.poses).copy()
            # OpenCV to OpenGL coordinate system conversion
            pose[:, 1:3] *= -1
            self._camera.matrix = pose
            fx, fy, cx, cy = cam.intrinsics
            w, h = cam.image_sizes
            main_cam_node = self._scene.main_camera_node
            assert main_cam_node is not None, "Main camera node is missing"
            main_cam = main_cam_node.camera
            assert main_cam is not None, "Main camera is missing"
            main_cam.fx, main_cam.fy, main_cam.cx, main_cam.cy = fx, fy, cx, cy
            self._renderer.viewport_width, self._renderer.viewport_height = w, h
            out = self._renderer.render(self._scene, flags)
            assert out is not None, "Rendering failed"
            color, depth = out
            if cam.nears_fars is not None:
                main_cam.znear, main_cam.zfar = cam.nears_fars
            # Fix depth 0 == far
            depth = np.where(depth == 0, main_cam.zfar, depth)
            return {
                "color": color,
                "depth": depth,
            }
        finally:
            self._platform.make_uncurrent()

    def close(self):
        if self._platform is not None:
            self._platform.make_current()
        if self._renderer is not None:
            self._renderer.delete()
            del self._renderer
            self._renderer = None
        if self._platform is not None:
            self._platform.delete_context()
            del self._platform
            self._platform = None
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None
        import gc
        gc.collect()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def export_mesh(self, path: str, **kwargs):
        del kwargs
        os.makedirs(path, exist_ok=True)
        mesh_path = os.path.join(self._model_path, "mesh.ply")
        shutil.copy(mesh_path, os.path.join(path, "mesh.ply"))
        logger.info(f"Mesh exported to {path}/mesh.ply")
