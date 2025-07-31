import logging
import pprint
import dataclasses
import json
import io
import base64
import yaml
from contextlib import contextmanager
import warnings
from functools import partial
import numpy as np
import argparse
import os
import ast
import importlib.util
from typing import cast, Optional, List
from nerfbaselines import Method, MethodInfo, ModelInfo, Dataset, OptimizeEmbeddingOutput
from nerfbaselines.utils import pad_poses, image_to_srgb, convert_image_dtype
from typing import Optional, Union
from typing_extensions import Literal, get_origin, get_args

import torch  # type: ignore
from torch.nn import functional as F  # type: ignore


def numpy_to_base64(array: np.ndarray) -> str:
    with io.BytesIO() as f:
        np.save(f, array)
        return base64.b64encode(f.getvalue()).decode("ascii")


def numpy_from_base64(data: str) -> np.ndarray:
    with io.BytesIO(base64.b64decode(data)) as f:
        return np.load(f)


def cast_value(tp, value):
    origin = get_origin(tp)
    if origin is Literal:
        for val in get_args(tp):
            try:
                value_casted = cast_value(type(val), value)
                if val == value_casted:
                    return value_casted
            except ValueError:
                pass
            except TypeError:
                pass
        raise TypeError(f"Value {value} is not in {get_args(tp)}")
    if origin is tuple:
        # NOTE: not supporting ellipsis
        if Ellipsis in get_args(tp):
            raise TypeError("Ellipsis not supported")
        if isinstance(value, str):
            value = value.split(",")
        if len(get_args(tp)) != len(value):
            raise TypeError(f"Length of value {value} is not equal to {tp}")
        return tuple(cast_value(t, v) for t, v in zip(get_args(tp), value))
    if origin is Union:
        for t in get_args(tp):
            try:
                return cast_value(t, value)
            except ValueError:
                import traceback
                traceback.print_exc()
                pass
            except TypeError:
                import traceback
                traceback.print_exc()
                pass
        raise TypeError(f"Value {value} is not in {tp}")
    if tp is type(None):
        if str(value).lower() == "none":
            return None
        else:
            raise TypeError(f"Value {value} is not None")
    if tp is bool:
        if str(value).lower() in {"true", "1", "yes"}:
            return True
        elif str(value).lower() in {"false", "0", "no"}:
            return False
        else:
            raise TypeError(f"Value {value} is not a bool")
    if tp in {int, float, bool, str}:
        return tp(value)
    if isinstance(value, tp):
        return value
    raise TypeError(f"Cannot cast value {value} to type {tp}")


def flatten_hparams(hparams, *, separator: str = "/", _prefix: str = ""):
    def format_value(v, only_simple_types=True):
        if isinstance(v, (str, float, int, bool, bytes, type(None))):
            return v
        if torch.is_tensor(v) or isinstance(v, np.ndarray):
            return format_value(v.tolist(), only_simple_types=only_simple_types)
        if isinstance(v, (list, tuple)):
            # If list of simple types, convert to string
            if not only_simple_types:
                return type(v)([format_value(x, only_simple_types=False) for x in v])
            formatted = [format_value(x) for x in v]
            if all(isinstance(x, (str, float, int, bool, bytes, type(None))) for x in formatted):
                return ",".join(str(x) for x in formatted)
            return ",".join(pprint.pformat(x) for x in formatted)
        if isinstance(v, dict):
            if not only_simple_types:
                return {k: format_value(v, only_simple_types=False) for k, v in v.items()}
            return pprint.pformat(format_value(v, only_simple_types=False))
        if isinstance(v, type):
            return v.__module__ + ":" + v.__name__
        if dataclasses.is_dataclass(v):
            return format_value({
                f.name: getattr(v, f.name) for f in dataclasses.fields(v)
            }, only_simple_types=only_simple_types)
        if callable(v):
            return v.__module__ + ":" + v.__name__
        return repr(v)

    flat = {}
    if dataclasses.is_dataclass(hparams):
        hparams = {f.name: getattr(hparams, f.name) for f in dataclasses.fields(hparams)}
    _blacklist = set((
        "port",
        "ckpt",
        "disable_viewer",
        "render_traj_path",
        "data_dir",
        "result_dir",
        "lpips_net",
        "tb_every",
        "tb_save_image",
    ))
    for k, v in hparams.items():
        if _prefix:
            k = f"{_prefix}{separator}{k}"
        if k in _blacklist:
            continue
        if isinstance(v, dict) or dataclasses.is_dataclass(v):
            flat.update(flatten_hparams(v, _prefix=k, separator=separator))
        else:
            flat[k] = format_value(v)
    return flat


class gs_Parser:
    def __init__(self, 
                 data_dir: str,
                 factor: int = 1,
                 normalize: bool = False,
                 test_every: int = 8,
                 state = None,
                 dataset: Optional[Dataset] = None):
        assert factor == 1, "Factor must be 1"
        del test_every, data_dir

        if state is not None:
            self.transform = numpy_from_base64(state["transform_base64"])
            self.scene_scale = state["scene_scale"]
            self.points = self.points_rgb = None
            if state["num_points"] is not None:
                self.points = np.zeros((state["num_points"], 3), dtype=np.float32)
                self.points_rgb = np.zeros((state["num_points"], 3), dtype=np.uint8)
            self.num_train_images = state["num_train_images"]
            self.dataset = None
            return

        assert dataset is not None, "Dataset must be provided"
        self.num_train_images = len(dataset.get("images"))

        # Optional normalize
        from datasets.normalize import (  # type: ignore
            similarity_from_cameras, transform_cameras, transform_points, align_principle_axes
        )
        if normalize:
            points = dataset.get("points3D_xyz")
            camtoworlds = pad_poses(dataset.get("cameras").poses)
            transform = T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            if points is not None:
                points = transform_points(T1, points)

                T2 = align_principle_axes(points)
                camtoworlds = transform_cameras(T2, camtoworlds)
                points = transform_points(T2, points)

                transform = cast(np.ndarray, T2 @ T1)

            # Apply transform to the dataset
            dataset = dataset.copy()
            dataset["cameras"] = dataset["cameras"].replace(poses=camtoworlds[..., :3, :4])
            dataset["points3D_xyz"] = points
        else:
            transform = np.eye(4)
        self.transform = transform

        # size of the scene measured by cameras
        camera_locations = dataset["cameras"].poses[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)

        self.scene_scale = np.max(dists)
        self.points = dataset.get("points3D_xyz")
        self.points_rgb = dataset.get("points3D_rgb")
        self.dataset = dataset

    def export(self):
        return {
            "scene_scale": self.scene_scale,
            "num_points": len(self.points) if self.points is not None else None,
            "transform": self.transform.tolist(),
            "transform_base64": numpy_to_base64(self.transform),
            "num_train_images": self.num_train_images,
        }


class gs_Dataset:
    def __init__(self, parser, 
                 split: str = "train",
                 patch_size: Optional[int] = None,
                 load_depths: bool = False):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        self.dataset = self.preprocess_images(parser.dataset)

    def __len__(self):
        return self.parser.num_train_images

    @staticmethod
    def preprocess_images(dataset):
        if dataset is None:
            return dataset
        background_color = dataset["metadata"].get("background_color", None)
        dataset = dataset.copy()
        dataset["images"] = [image_to_srgb(image, 
                                           background_color=background_color, 
                                           dtype=np.uint8) for image in dataset["images"]]
        return dataset

    def __getitem__(self, idx):
        dataset = self.dataset
        image = dataset["images"][idx]
        fx, fy, cx, cy = dataset["cameras"][idx].intrinsics
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        camtoworlds = pad_poses(dataset["cameras"][idx].poses)
        mask = None
        if dataset.get("masks", None) is not None:
            mask = dataset["masks"][idx]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            if mask is not None:
                mask = mask[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": idx,  # the index of the image in the dataset
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(convert_image_dtype(mask, 'float32'))

        if self.load_depths:
            # projected points to image plane to get depths
            worldtocams = np.linalg.inv(camtoworlds)
            points_world = dataset["points3D_xyz"][dataset["images_points3D_indides"][idx]]
            points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
            points_proj = (K @ points_cam.T).T
            points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
            depths = points_cam[:, 2]  # (M,)
            # filter out points outside the image
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            points = points[selector]
            depths = depths[selector]
            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()
        return data


# Extract code dynamically
def _build_simple_trainer_module():
    module_spec = importlib.util.find_spec("simple_trainer", __package__)
    assert module_spec is not None and module_spec.origin is not None, "Failed to find simple_trainer module"
    with open(module_spec.origin, "r") as f:
        simple_trainer_ast = ast.parse(f.read())

    # Transform simple trainer, set num_workers=0
    class _Visitor(ast.NodeVisitor):
        def visit(self, node):
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "torch.utils.data.DataLoader":
                num_workers = next(x for x in node.keywords if x.arg == "num_workers")
                num_workers.value = ast.Constant(value=0, kind=None, lineno=0, col_offset=0)
                persistent_workers = next((x for x in node.keywords if x.arg == "persistent_workers"), None)
                if persistent_workers is not None:
                    node.keywords.remove(persistent_workers)
            super().visit(node)
    _Visitor().visit(simple_trainer_ast)

    # Filter imports
    simple_trainer_ast.body.remove(next(x for x in simple_trainer_ast.body if ast.unparse(x) == "from torch.utils.tensorboard import SummaryWriter"))
    simple_trainer_ast.body.remove(next(x for x in simple_trainer_ast.body if ast.unparse(x) == "import viser"))
    # simple_trainer_ast.body.remove(next(x for x in simple_trainer_ast.body if ast.unparse(x) == "import nerfview"))

    # Update config
    config_ast = next(x for x in simple_trainer_ast.body if getattr(x, "name", None) == "Config")
    assert isinstance(config_ast, ast.ClassDef)
    config_ast.body.insert(-1, ast.AnnAssign(
        target=ast.Name(id="app_test_opt_steps", ctx=ast.Store(), lineno=0, col_offset=0),
        annotation=ast.Name(id="int", ctx=ast.Load(), lineno=0, col_offset=0),
        value=ast.Constant(value=128, kind=None, lineno=0, col_offset=0), 
        simple=True, col_offset=0, lineno=0))
    config_ast.body.insert(-1, ast.AnnAssign(
        target=ast.Name(id="app_test_opt_lr", ctx=ast.Store(), lineno=0, col_offset=0),
        annotation=ast.Name(id="float", ctx=ast.Load(), lineno=0, col_offset=0),
        value=ast.Constant(value=0.1, kind=None, lineno=0, col_offset=0), 
        simple=True, col_offset=0, lineno=0))
    config_ast.body.insert(-1, ast.AnnAssign(  # type: ignore
        target=ast.Name(id="background_color", ctx=ast.Store(), lineno=0, col_offset=0),
        annotation=ast.parse("Optional[Tuple[float, float, float]]").body[0].value,  # type: ignore
        value=ast.Constant(value=None, kind=None, lineno=0, col_offset=0), 
        simple=True, col_offset=0, lineno=0))

    runner_ast = next(x for x in simple_trainer_ast.body if getattr(x, "name", None) == "Runner")
    assert isinstance(runner_ast, ast.ClassDef)
    runner_train_ast = next(x for x in runner_ast.body if getattr(x, "name", None) == "train")
    assert isinstance(runner_train_ast, ast.FunctionDef)
    runner_train_ast.name = "setup_train"
    # Training loop
    assert isinstance(runner_train_ast.body[-1], ast.For)
    # Train init body - we remove unused code
    init_train_body = list(runner_train_ast.body[:-3])
    init_train_body.pop(4)
    init_train_body.extend(ast.parse("""
self.trainloader=trainloader
self.trainloader_iter=trainloader_iter
self.schedulers=schedulers
""").body)
    iter_step_body = []
    iter_step_body.extend(ast.parse("""
trainloader_iter=self.trainloader_iter
trainloader=self.trainloader
schedulers=self.schedulers
""").body)
    iter_step_body.extend(init_train_body[:4])
    iter_step_body.extend((runner_train_ast.body[-1].body)[1:-1])
    iter_step_body.pop(-10)  # Remove write to tensorboard step
    save_step = iter_step_body.pop(-9)  # Remove save() step
    iter_step_body.pop(-2)  # Remove eval() step
    # Remove pbar.set_description
    iter_step_body.pop(next(i for i, step in enumerate(iter_step_body) if ast.unparse(step) == "pbar.set_description(desc)"))

    # NOTE: extend gsplat to use masks
    render_step_idx = next(i for i, step in enumerate(iter_step_body) if ast.unparse(step).startswith("(renders, alphas, info) = self.rasterize_splats"))
    iter_step_body.insert(render_step_idx + 1, ast.parse("""if data.get("mask") is not None:
    mask = data["mask"].to(self.device)
    renders = renders * mask + renders.detach() * (1 - mask)
    alphas = alphas * mask + alphas.detach() * (1 - mask)
""").body[0])
    
    bkgd_blend_step_idx = next(i for i, step in enumerate(iter_step_body) if ast.unparse(step).startswith("if cfg.random_bkgd:"))
    iter_step_body.insert(bkgd_blend_step_idx + 1, ast.parse("""if not cfg.random_bkgd and cfg.background_color is not None:
    bkgd = torch.tensor(cfg.background_color, dtype=colors.dtype, device=self.device).view(1, 1, 1, 3)
    colors = colors + bkgd * (1.0 - alphas)
""").body[0])

    iter_step_body.extend(ast.parse("""def _():
    self.trainloader_iter=trainloader_iter
    out={"loss": loss.item(), "l1loss": l1loss.item(), "ssim": ssimloss.item(), "num_gaussians": len(self.splats["means"])}
    if cfg.depth_loss:
        out["depthloss"] = depthloss.item()
    return out
""").body[0].body)  # type: ignore
    runner_train_ast.body = init_train_body
    runner_ast.body.append(ast.FunctionDef(lineno=0, col_offset=0,
        name="train_iteration",
        args=ast.arguments(  # type: ignore
            args=[
                ast.arg(arg="self", annotation=None, lineno=0, col_offset=0),
                ast.arg(arg="step", annotation=None, lineno=0, col_offset=0),
            ],
            posonlyargs=[], kwonlyargs=[], kw_defaults=[], defaults=[],
            kwarg=None, kwargannotation=None, return_annotation=None
        ),
        body=iter_step_body, decorator_list=[],
    ))

    # Save method
    save_step_body = save_step.body[4:]  # Strip saving stats
    save_step_body.insert(0, ast.parse("cfg=self.cfg").body[0])
    save_step_body.insert(0, ast.parse("world_size=self.world_size").body[0])
    # Change saving location
    save_step_body[-1].value.args[1].values[0].value = ast.Name(id="path", ctx=ast.Load(), lineno=0, col_offset=0)
    runner_ast.body.append(ast.FunctionDef(lineno=0, col_offset=0,
        name="save",
        args=ast.arguments(  # type: ignore
            args=[
                ast.arg(arg="self", annotation=None, lineno=0, col_offset=0),
                ast.arg(arg="step", annotation=None, lineno=0, col_offset=0),
                ast.arg(arg="path", annotation=None, lineno=0, col_offset=0),
            ],
            posonlyargs=[], kwonlyargs=[], kw_defaults=[], defaults=[],
            kwarg=None, kwargannotation=None, return_annotation=None
        ),
        body=save_step_body, decorator_list=[],
    ))

    # Init method
    init_method = next(x for x in runner_ast.body if getattr(x, "name", None) == "__init__")
    assert isinstance(init_method, ast.FunctionDef)
    init_method.body = init_method.body[:6] + init_method.body[14:-2]  # Remove unused code
    init_method.args.args.append(ast.arg(arg="Parser", annotation=None, lineno=0, col_offset=0))
    init_method.args.args.append(ast.arg(arg="Dataset", annotation=None, lineno=0, col_offset=0))

    # Execute code to build module
    module = {}
    exec(compile(simple_trainer_ast, "<string>", "exec"), module)
    return argparse.Namespace(**module)


class GSplat(Method):
    def __init__(self, *, train_dataset=None, checkpoint=None, config_overrides=None):
        super().__init__()

        self.checkpoint = checkpoint
        self.simple_trainer = _build_simple_trainer_module()

        # Build trainer
        self.cfg = self._get_config(checkpoint, config_overrides)

        # Load parser state
        parser_state = None
        if checkpoint is not None and os.path.exists(f"{checkpoint}/parser.json"):
            with open(f"{checkpoint}/parser.json", "r") as f:
                parser_state = json.load(f)

        # Build runner
        local_rank = world_rank = 0
        world_size = 1
        self.runner = self.simple_trainer.Runner(
            local_rank, world_rank, world_size, self.cfg, Dataset=gs_Dataset, Parser=partial(gs_Parser, dataset=train_dataset, state=parser_state))
        self.step = 0
        self._loaded_step = None

        # Load checkpoint if available
        if checkpoint is not None:
            ckpt_files = [os.path.join(checkpoint, x) for x in os.listdir(checkpoint) if x.startswith("ckpt_") and x.endswith(".pt")]
            ckpt_files.sort(key=lambda x: int(x.split("_rank")[-1].split(".")[0]))
            ckpts = [
                torch.load(file, map_location=self.runner.device, weights_only=True)
                for file in ckpt_files
            ]
            for k in self.runner.splats.keys():
                feat = torch.cat([ckpt["splats"][k] for ckpt in ckpts]) if len(ckpts) > 1 else ckpts[0]["splats"][k]
                self.runner.splats[k].data = feat
            if self.cfg.pose_opt:
                self.runner.pose_adjust.load_state_dict(ckpts[0]["pose_adjust"])
            if self.cfg.app_opt:
                self.runner.app_module.load_state_dict(ckpts[0]["app_module"])
            self.step = self._loaded_step = ckpts[0]["step"]

        # Setup dataloaders if training mode
        if train_dataset is not None:
            self.runner.setup_train()

    def _get_config(self, checkpoint, config_overrides):
        cfg = self.simple_trainer.Config(
            strategy=self.simple_trainer.DefaultStrategy(verbose=True),
        )
        cfg.data_factor = 1

        if checkpoint is not None:
            with open(f"{checkpoint}/cfg.yml", "r") as f:
                cfg_dict = yaml.load(f, Loader=yaml.UnsafeLoader)
            cfg.__dict__.update(cfg_dict)

        # Apply config overrides
        field_types = {k.name: k.type for k in dataclasses.fields(cfg)}
        if "strategy" in (config_overrides or {}):
            import gsplat.strategy as strategies  # type: ignore
            cfg.strategy = getattr(strategies, config_overrides["strategy"])()
        strat_types = {k.name: k.type for k in dataclasses.fields(cfg.strategy)}
        for k, v in (config_overrides or {}).items():
            if k == "strategy":
                continue
            if k.startswith("strategy."):
                v = cast_value(strat_types[k], v)
                setattr(cfg.strategy, k[len("strategy."):], v)
                continue
            v = cast_value(field_types[k], v)
            setattr(cfg, k, v)

        cfg.adjust_steps(cfg.steps_scaler)
        cfg.steps_scaler = 1.0  # We have already adjusted the steps
        if cfg.pose_opt:
            warnings.warn("Pose optimization is enabled, but it will only by applied to training images. No test-time pose optimization is enabled.")
        return cfg

    @classmethod
    def get_method_info(cls):
        return MethodInfo(
            method_id="",  # Will be filled in by the registry
            required_features=frozenset(("points3D_xyz", "points3D_rgb", "color", "images_points3D_indices")),
            supported_camera_models=frozenset(("pinhole",)),
            supported_outputs=("color", "depth", "accumulation"),
            viewer_default_resolution=768,
        )

    def get_info(self):
        return ModelInfo(
            **self.get_method_info(),
            hparams=flatten_hparams(self.cfg),
            num_iterations=self.cfg.max_steps,
            loaded_step=self._loaded_step,
            loaded_checkpoint=self.checkpoint,
        )

    def train_iteration(self, step):
        self.step = step
        out = self.runner.train_iteration(step)
        self.step = step+1
        return out

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/cfg.yml", "w") as f:
            yaml.dump(vars(self.cfg), f)
        with open(f"{path}/parser.json", "w") as f:
            json.dump(self.runner.parser.export(), f)
        self.runner.save(self.step, path)

    @contextmanager
    def _patch_embedding(self, embedding=None):
        if embedding is None:
            yield None
            return
        embeds_module = None
        try:
            was_called = False
            if self.cfg.app_opt:
                embeds_module = self.runner.app_module.embeds
                class _embed(torch.nn.Module):
                    def forward(*args, **kwargs):
                        del args, kwargs
                        nonlocal was_called
                        was_called = True
                        return embedding[None]
                self.runner.app_module.embeds = _embed()
            yield None
            if self.cfg.app_opt:
                assert was_called, "Failed to patch appearance embedding"
        finally:
            # Return embeds back
            if embeds_module is not None:
                self.runner.app_module.embeds = embeds_module

    def _add_background_color(self, img, accumulation):
        if self.cfg.background_color is None:
            return img
        background_color = torch.tensor(self.cfg.background_color, device=img.device, dtype=torch.float32).view(1, 1, 1, 3)
        return img + (1.0 - accumulation) * background_color

    def _format_output(self, output, options):
        del options
        return {
            k: v.cpu().numpy() for k, v in output.items()
        }

    @torch.no_grad()
    def render(self, camera, *, options=None):
        camera = camera.item()
        from datasets.normalize import transform_cameras  # type: ignore
        cfg = self.cfg
        device = self.runner.device
        camtoworlds_np = transform_cameras(self.runner.parser.transform, pad_poses(camera.poses[None]))
        camtoworlds = torch.from_numpy(camtoworlds_np).float().to(device)
        fx, fy, cx, cy = camera.intrinsics
        Ks = torch.from_numpy(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])).float().to(device)
        width, height = camera.image_sizes

        # Patch appearance
        embedding_np = (options or {}).get("embedding")
        embedding = torch.from_numpy(embedding_np).to(self.runner.device) if embedding_np is not None else None
        outputs = (options or {}).get("outputs") or ()
        with self._patch_embedding(embedding):
            colors, accumulation, _ = self.runner.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks[None],
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=object() if embedding is not None else None,
                render_mode="RGB+ED" if "depth" in outputs else "RGB",
            )  # [1, H, W, 3]
        color = self._add_background_color(colors[..., :3], accumulation)
        out = {
            "color": torch.clamp(color.squeeze(0), 0.0, 1.0).detach(),
            "accumulation": accumulation.squeeze(0).squeeze(-1).detach(),
        }
        if colors.shape[-1] > 3:
            out["depth"] = colors.squeeze(0)[..., 3].detach()
        return self._format_output(out, options)

    def get_train_embedding(self, index):
        if not self.cfg.app_opt:
            return None
        return self.runner.app_module.embeds.weight[index].detach().cpu().numpy()

    def optimize_embedding(self, dataset: Dataset, *, embedding=None) -> OptimizeEmbeddingOutput:
        if not self.cfg.app_opt:
            raise NotImplementedError("Appearance optimization is not enabled, add --set app_opt=True to the command line.")
        assert len(dataset["images"]) == 1, "Only single image optimization is supported"
        camera = dataset["cameras"].item()
        from datasets.normalize import transform_cameras  # type: ignore
        cfg = self.cfg
        device = self.runner.device
        camtoworlds_np = transform_cameras(self.runner.parser.transform, pad_poses(camera.poses[None]))
        camtoworlds = torch.from_numpy(camtoworlds_np).float().to(device)
        fx, fy, cx, cy = camera.intrinsics
        Ks = torch.from_numpy(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])).float().to(device)
        width, height = camera.image_sizes
        dataset = gs_Dataset.preprocess_images(dataset)
        pixels = torch.from_numpy(dataset["images"][0]).float().to(device).div(255.)
        # Extend gsplat, handle masks
        masks = None
        _dataset_masks = dataset.get("masks")
        if _dataset_masks is not None:
            masks = torch.from_numpy(convert_image_dtype(_dataset_masks[0], np.float32)).to(device)[None]

        # Patch appearance
        if embedding is not None:
            embedding_th = torch.from_numpy(embedding).to(self.runner.device)
        else:
            embedding_th = torch.zeros_like(self.runner.app_module.embeds.weight[0])
        embedding_th = torch.nn.Parameter(embedding_th.requires_grad_())
        optimizer = torch.optim.Adam([embedding_th], lr=self.cfg.app_test_opt_lr)
        l1losses: List[float] = []
        ssimlosses: List[float] = []
        losses: List[float] = []
        with torch.enable_grad(), self._patch_embedding(embedding_th):
            for _ in range(self.cfg.app_test_opt_steps):
                colors, accumulation, _ = self.runner.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks[None],
                    width=width,
                    height=height,
                    sh_degree=cfg.sh_degree,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    image_ids=object()
                )  # [1, H, W, 3]
                colors = self._add_background_color(colors, accumulation)

                # Scale colors grad by masks
                if masks is not None:
                    colors = colors * masks + colors.detach() * (1.0 - masks)

                l1loss = F.l1_loss(colors, pixels[None])
                ssimloss = 1.0 - self.runner.ssim(
                    pixels[None].permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
                )
                loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
                l1losses.append(l1loss.item())
                ssimlosses.append(ssimloss.item())
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return {
            "metrics": {
                "loss": losses,
                "l1loss": l1losses,
                "ssim": ssimlosses,
            },
            "embedding": embedding_th.detach().cpu().numpy(),
        }

    def export_gaussian_splats(self, *, options=None):
        from nerfbaselines.utils import invert_transform

        options = options or {}
        dataset_metadata = options.get("dataset_metadata") or {}
        splats = self.runner.splats

        if self.cfg.app_opt:
            from nerfbaselines.utils import apply_transform, invert_transform
            if "viewer_transform" in dataset_metadata and "viewer_initial_pose" in dataset_metadata:
                viewer_initial_pose_ws = apply_transform(invert_transform(dataset_metadata["viewer_transform"], has_scale=True), dataset_metadata["viewer_initial_pose"])
                camera_center = torch.tensor(viewer_initial_pose_ws[:3, 3], dtype=torch.float32, device="cuda")
            else:
                camera_center = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
            logging.warning("gsplat does not support view-dependent demo when appearance is enabled (app_opt=True). We will bake the appearance of a single appearance embedding and single viewing direction.")
            embedding_np = (options or {}).get("embedding")
            embedding = torch.from_numpy(embedding_np).to(self.runner.device) if embedding_np is not None else None
            with torch.no_grad(), self._patch_embedding(embedding):
                # Get the embedding
                colors = self.runner.app_module(
                    features=splats["features"],
                    embed_ids=torch.zeros((1,), dtype=torch.long, device="cuda"),
                    dirs=splats["means"][None, :, :] - camera_center[None, None, :],
                    sh_degree=self.cfg.sh_degree,
                )
                colors = colors + splats["colors"]
                colors = torch.sigmoid(colors).squeeze(0)[..., None]
                assert len(colors.shape) == 3 and colors.shape[1:] == (3, 1), f"Invalid colors shape {colors.shape}"
                # Convert to spherical harmonics of deg 0
                C0 = 0.28209479177387814
                spherical_harmonics = (colors - 0.5) / C0
        else:
            spherical_harmonics = torch.cat((splats["sh0"], splats["shN"]), dim=1).transpose(1, 2)

        out = dict(
            means=splats["means"].detach().cpu().numpy(),
            scales=splats["scales"].exp().detach().cpu().numpy(),
            opacities=torch.nn.functional.sigmoid(splats["opacities"]).detach().cpu().numpy(),
            quaternions=torch.nn.functional.normalize(splats["quats"]).detach().cpu().numpy(),
            spherical_harmonics=spherical_harmonics.detach().cpu().numpy())
        if self.cfg.antialiased:
            out["antialias_2D_kernel_size"] = 0.3

        # Add transform params
        options = options or {}
        if self.runner.parser.transform is not None:
            transform = self.runner.parser.transform.copy()
            inv_transform = invert_transform(transform, has_scale=True)
            out["transform"] = inv_transform[:3, :4]
        return out
