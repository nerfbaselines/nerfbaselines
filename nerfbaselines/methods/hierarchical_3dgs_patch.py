import itertools
import copy
import ast
from typing import cast
from ._patching import Context

import_context = Context()

metrics = {
    "loss": "loss.item()",
    "l1_loss": "Ll1.item()",
    "l1_depth": "Ll1depth_pure.item() if hasattr(Ll1depth_pure, 'item') else Ll1depth_pure",
    "ssim": "(1 - Lssim.item()) if 'Lssim' in locals() else (1 - (loss.item() - (1.0 - self._opt.lambda_dssim) * Ll1.item()) / self._opt.lambda_dssim)",
    "psnr": "10 * torch.log10(torch.mean(locals().get('alpha_mask', torch.tensor(1.0))) / torch.mean((image - gt_image) ** 2 * locals().get('alpha_mask', 1))).item()",
    "num_points": "len(gaussians.get_xyz)",
}
train_step_disabled_names = [
    "prepare_output_and_logger",
    "debug_from",
    "iter_start", 
    "iter_end",
    "training_report",
    "network_gui", 
    "checkpoint_iterations",
    "saving_iterations",
    "loss_dict",
    "tb_writer",
    "progress_bar",
    "ema_loss_for_log",
    "ema_dist_for_log",
    "ema_normal_for_log",
    "ema_Ll1depth_for_log",
]

def _ast_prune_node(tree, callback):
    tree = copy.deepcopy(tree)
    sentinel = ast.AST(lineno=0, col_offset=0)
    if isinstance(tree, list):
        return [x for x in (_ast_prune_node(x, callback) for x in tree) if x is not None]
    class Transformer(ast.NodeTransformer):
        def visit(self, node):
            if callback(node):
                return sentinel
            out = self.generic_visit(node)
            for k, v in ast.iter_fields(out):
                if v == sentinel:
                    return sentinel
                if isinstance(v, list):
                    values = [x for x in v if x is not sentinel]
                    if v and not values:
                        return sentinel
                    setattr(out, k, values)
            return out
    out = Transformer().visit(tree)
    if out is sentinel:
        return None
    return out


def ast_remove_names(tree, names):
    return _ast_prune_node(tree, lambda node: isinstance(node, ast.Name) and node.id in names)


def single(xs):
    out = ()
    for x in xs:
        if len(out) == 1:
            raise ValueError("Expected single element, got more")
        out = (x,)
    if not out:
        raise ValueError("Expected single element, got none")
    return out[0]


@import_context.patch_ast_import("scene")
def _(ast_module: ast.Module):
    # Find Scene class
    Scene_ast = single((x for x in ast_module.body if isinstance(x, ast.ClassDef) and x.name == "Scene"))
    Scene_init_ast = single((x for x in Scene_ast.body if isinstance(x, ast.FunctionDef) and x.name == "__init__"))
    # Remove 'if not self.loaded_iter:'
    if_instance = single((x for x in Scene_init_ast.body if isinstance(x, ast.If) and isinstance(x.test, ast.UnaryOp) and isinstance(x.test.op, ast.Not) and isinstance(x.test.operand, ast.Attribute) and x.test.operand.attr == "loaded_iter"))
    Scene_init_ast.body.remove(if_instance)
    # Remove 'if os.path.exists(os.path.join(args.source_path, "sparse")):'
    if_instance = single((x for x in Scene_init_ast.body if isinstance(x, ast.If) and isinstance(x.test, ast.Call) and isinstance(x.test.func, ast.Attribute) and x.test.func.attr == "exists"))
    Scene_init_ast.body.remove(if_instance)
    # Add scene_info as argument to Scene
    Scene_init_ast.args.args.insert(1, ast.arg(arg="scene_info", annotation=None, lineno=0, col_offset=0))


# Patch train to extract the training loop and init
# <patch train>
@import_context.patch_ast_import("train_coarse")
@import_context.patch_ast_import("train_post")
@import_context.patch_ast_import("train_single")
def _(ast_module: ast.Module):
    training_ast = copy.deepcopy(next(x for x in ast_module.body if isinstance(x, ast.FunctionDef) and x.name == "training"))
    # We remove the unused code
    training_ast = ast_remove_names(training_ast, train_step_disabled_names)

    # Now, we extract the train iteration code, inside of the following code:
    # while iteration < opt.iterations + 1:
    #     for viewpoint_batch in training_generator:
    #         for viewpoint_cam in viewpoint_batch:
    train_loop = next((x for x in training_ast.body if isinstance(x, ast.While) and isinstance(x.test, ast.Compare)), None)
    assert train_loop is not None, "Could not find training loop"
    train_step = train_loop.body[0]
    train_step = train_step.body[0].body
    # Add return statement to train_step
    _metrics = metrics.copy()
    # If Ll1depth_pure is not defined, remove l1_depth from metrics
    has_l1_depth = False
    for node in itertools.chain(*(ast.walk(x) for x in train_step)):
        if isinstance(node, ast.Name) and node.id == "Ll1depth_pure":
            has_l1_depth = True
            break
    if not has_l1_depth:
        _metrics.pop("l1_depth")
    train_step.append(ast.Return(value=ast.Dict(
        keys=[ast.Constant(value=name, kind=None, lineno=0, col_offset=0) for name in _metrics.keys()], 
        values=[ast.parse(value).body[0].value for value in _metrics.values()],  # type: ignore
        lineno=0, col_offset=0), lineno=0, col_offset=0))
    
    # Remove 'if iteration % 10 == 0'
    train_step = _ast_prune_node(train_step, 
        lambda node: isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and isinstance(node.test.ops[0], ast.Eq) and
        isinstance(node.test.comparators[0], ast.Constant) and node.test.comparators[0].value == 0 and
        isinstance(node.test.left, ast.BinOp) and isinstance(node.test.left.op, ast.Mod) and isinstance(node.test.left.right, ast.Constant) and node.test.left.right.value == 10)
    # Remove 'if iteration == self._opt.iterations:'
    train_step = _ast_prune_node(train_step,
        lambda node: isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and isinstance(node.test.ops[0], ast.Eq) and isinstance(node.test.comparators[0], ast.Attribute) and node.test.comparators[0].attr == "iterations" and node.test.comparators[0].value.id == "opt")
    # If there is no background = ..., add it
    if not any(isinstance(x, ast.Assign) and isinstance(x.targets[0], ast.Name) and x.targets[0].id == "background" for x in train_step):
        value = ast.parse("torch.tensor(self._bg_color, dtype=torch.float32, device='cuda')").body[0].value
        train_step.insert(0, ast.Assign(
            targets=[ast.Name(id="background", ctx=ast.Store(lineno=0, col_offset=0), lineno=0, col_offset=0)],
            value=value,
            lineno=0, col_offset=0))

    # Detect global names
    setup_body = [x for x in training_ast.body if x != train_loop]
    global_names = set([x.arg for x in training_ast.args.args])
    for instruction in setup_body:
        for node in ast.walk(instruction):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                global_names.add(node.id)
    global_names.remove("iteration")
    global_names.remove("background")
    global_names.add("args")

    # Replace all global names with `self`
    class Transformer(ast.NodeTransformer):
        def __init__(self):
            self._stored_names = set()
        def visit_Name(self, node):
            if node.id in global_names:
                if isinstance(node.ctx, ast.Store):
                    self._stored_names.add(node.id)
                return ast.copy_location(
                    ast.Attribute(value=ast.Name(id="self", ctx=ast.Load(lineno=0), lineno=0, col_offset=0), 
                                  attr="_"+node.id, ctx=node.ctx), node)
            return self.generic_visit(node)

    train_step_transformer = Transformer()
    for instruction in train_step:
        train_step_transformer.visit(instruction)

    # Define function train_iteration
    train_iteration = cast(ast.FunctionDef, ast.parse("def train_iteration(self, viewpoint_cam, iteration):\n    pass").body[0])
    train_iteration.body = train_step
    # Add `return metrics` where metrics is obtained as {name: eval(name) for name in metrics}
    ast_module.body.append(train_iteration)

    # Now, we setup get_argparser function
    main_body = next(x.body for x in ast_module.body if isinstance(x, ast.If) and getattr(getattr(x.test, 'left'), 'id') == "__name__")
    get_argparser_body = list(itertools.takewhile(lambda x: not isinstance(x, ast.Assign) or x.targets[0].id != "args", main_body))
    # Assign self._lp, self._op, self._pp
    for name in ["lp", "op", "pp"]:
        get_argparser_body.append(ast.Assign(
            targets=[ast.Attribute(
                value=ast.Name(id="self", ctx=ast.Load(lineno=0, col_offset=0), lineno=0, col_offset=0), 
                attr="_"+name, 
                ctx=ast.Store(lineno=0, col_offset=0),
                lineno=0, col_offset=0)],
            value=ast.parse(name).body[0].value, lineno=0, col_offset=0))
    get_argparser_body.append(ast.Return(value=ast.Name(id="parser", ctx=ast.Load(), lineno=0, col_offset=0), lineno=0, col_offset=0))
    get_argparser = cast(ast.FunctionDef, ast.parse("def get_argparser(self):\n    pass").body[0])
    get_argparser.body = get_argparser_body
    ast_module.body.append(get_argparser)

    # Build setup_train function
    setup_train = ast_remove_names([x for x in training_ast.body if x != train_loop], ["first_iter"])
    for instruction in setup_train:
        Transformer().visit(instruction)
    setup_train_start = list(itertools.dropwhile(lambda x: not isinstance(x, ast.Assign) or x.targets[0].id != "args", main_body))
    training_call = setup_train_start[-2]
    setup_train_start = setup_train_start[3:-2]
    assert isinstance(training_call, ast.Expr) and isinstance(training_call.value, ast.Call)
    param_names = [x.arg for x in training_ast.args.args]
    for name, value in zip(param_names, training_call.value.args):
        # Assign the value to self._{name}
        setup_train_start.append(ast.Assign(
            targets=[ast.Attribute(
                value=ast.Name(id="self", ctx=ast.Load(lineno=0, col_offset=0), lineno=0, col_offset=0), 
                attr="_"+name, 
                ctx=ast.Store(lineno=0, col_offset=0),
                lineno=0, col_offset=0)],
            value=value, lineno=0, col_offset=0))
    # Load lp, op, pp
    setup_train_start0 = []
    for name in ["lp", "op", "pp"]:
        setup_train_start0.append(ast.Assign(
            targets=[ast.Name(id=name, ctx=ast.Store(lineno=0), lineno=0, col_offset=0)],
            value=ast.Attribute(
                value=ast.Name(id="self", ctx=ast.Load(lineno=0), lineno=0, col_offset=0), 
                attr="_"+name, 
                ctx=ast.Load(lineno=0, col_offset=0),
                lineno=0, col_offset=0), lineno=0, col_offset=0))
    # Drop last two instructions which save state and print training finished
    setup_train = setup_train_start0 + setup_train_start + setup_train
    setup_train = ast_remove_names(setup_train, ["network_gui", "tqdm"])
    setup_train_function = cast(ast.FunctionDef, ast.parse("def setup_train(self, args, Scene):\n    pass").body[0])
    setup_train_function.body = setup_train
    ast_module.body.append(setup_train_function)

    # Use this code to debug when integrating new codebase
    # print("===== Setup train =====") 
    # print(ast.unparse(setup_train))
    # print()
    # print("===== Train step =====")
    # print(ast.unparse(train_step))
    # print()
    # print("===== Train step sets =====")
    # print(train_step_transformer._stored_names)
    # print()
    # print("===== Get argparser =====")
    # print(ast.unparse(get_argparser))


@import_context.patch_ast_import("preprocess.make_depth_scale")
def _(ast_module: ast.Module):
    # Remove imports
    # from joblib import delayed, Parallel
    # from read_write_model import *
    new_body = []
    for instruction in ast_module.body:
        if isinstance(instruction, ast.ImportFrom) and instruction.module == "joblib":
            continue
        if isinstance(instruction, ast.ImportFrom) and instruction.module == "read_write_model":
            continue
        new_body.append(instruction)
    ast_module.body = new_body
    get_scales_ast = single((x for x in ast_module.body if isinstance(x, ast.FunctionDef) and x.name == "get_scales"))
    # Add invmonodepthmap argument
    get_scales_ast.args.args.append(ast.arg(arg="invmonodepthmap", annotation=None, lineno=0, col_offset=0))
    get_scales_ast.args.args.append(ast.arg(arg="images_metas", annotation=None, lineno=0, col_offset=0))
    # Remove line invmonodepthmap = ... from the function body
    get_scales_ast.body = _ast_prune_node(get_scales_ast.body, lambda node: isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "invmonodepthmap")
    ast_module.body.insert(0, ast.parse("from nerfbaselines.datasets._colmap_utils import *").body[0])


import_context.apply_patch(r"""
diff --git a/preprocess/make_chunk.py b/preprocess/make_chunk.py
index 1eb0529..f575c95 100644
--- a/preprocess/make_chunk.py
+++ b/preprocess/make_chunk.py
@@ -12,11 +12,11 @@
 import numpy as np
 import argparse
 import cv2
-from joblib import delayed, Parallel
 import os
 import random
-from read_write_model import *
-import json
+from argparse import Namespace
+from nerfbaselines.datasets._colmap_utils import qvec2rotmat
+
 
 def get_nb_pts(image_metas):
     n_pts = 0
@@ -27,31 +27,21 @@ def get_nb_pts(image_metas):
 
     return n_pts + 1
 
-if __name__ == '__main__':
-    random.seed(0)
+
+def get_argparser():
     parser = argparse.ArgumentParser()
-    parser.add_argument('--base_dir', required=True)
-    parser.add_argument('--images_dir', required=True)
     parser.add_argument('--chunk_size', default=100, type=float)
     parser.add_argument('--min_padd', default=0.2, type=float)
     parser.add_argument('--lapla_thresh', default=1, type=float, help="Discard images if their laplacians are < mean - lapla_thresh * std") # 1
     parser.add_argument('--min_n_cams', default=100, type=int) # 100
     parser.add_argument('--max_n_cams', default=1500, type=int) # 1500
-    parser.add_argument('--output_path', required=True)
     parser.add_argument('--add_far_cams', default=True)
-    parser.add_argument('--model_type', default="bin")
+    return parser
 
-    args = parser.parse_args()
 
+def generate_chunks(cam_intrinsics, images_metas, points3d, images, args):
     # eval
-    test_file = f"{args.base_dir}/test.txt"
-    if os.path.exists(test_file):
-        with open(test_file, 'r') as file:
-            test_cam_names_list = file.readlines()
-            blending_dict = {name[:-1] if name[-1] == '\n' else name: {} for name in test_cam_names_list}
-
-    cam_intrinsics, images_metas, points3d = read_model(args.base_dir, ext=f".{args.model_type}")
-
+    random.seed(0)
     cam_centers = np.array([
         -qvec2rotmat(images_metas[key].qvec).astype(np.float32).T @ images_metas[key].tvec.astype(np.float32)
         for key in images_metas
@@ -108,18 +98,12 @@ if __name__ == '__main__':
     global_bbox[0, 2] = -1e12
     global_bbox[1, 2] = 1e12
 
-    def get_var_of_laplacian(key):
-        image = cv2.imread(os.path.join(args.images_dir, images_metas[key].name))
-        if image is not None:
-            gray = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY)
-            return cv2.Laplacian(gray, cv2.CV_32F).var()
-        else:
-            return 0   
+    def get_var_of_laplacian(i):
+        gray = cv2.cvtColor(images[i][..., :3], cv2.COLOR_RGB2GRAY)
+        return cv2.Laplacian(gray, cv2.CV_32F).var()
         
     if args.lapla_thresh > 0: 
-        laplacians = Parallel(n_jobs=-1, backend="threading")(
-            delayed(get_var_of_laplacian)(key) for key in images_metas
-        )
+        laplacians = [get_var_of_laplacian(i) for i in range(len(images_metas))]
         laplacians_dict = {key: laplacian for key, laplacian in zip(images_metas, laplacians)}
 
     excluded_chunks = []
@@ -200,17 +184,13 @@ if __name__ == '__main__':
             print(f"{valid_cam.sum()} after random removal")
 
         valid_keys = [key for idx, key in enumerate(images_metas) if valid_cam[idx]]
-        
-        if valid_cam.sum() > args.min_n_cams:# or init_valid_cam.sum() > 0:
-            out_path = os.path.join(args.output_path, f"{i}_{j}")
-            out_colmap = os.path.join(out_path, "sparse", "0")
-            os.makedirs(out_colmap, exist_ok=True)
 
+        if valid_cam.sum() > args.min_n_cams:# or init_valid_cam.sum() > 0:
             # must remove sfm points to use colmap triangulator in following steps
             images_out = {}
             for key in valid_keys:
                 image_meta = images_metas[key]
-                images_out[key] = Image(
+                images_out[key] = Namespace(
                     id = key,
                     qvec = image_meta.qvec,
                     tvec = image_meta.tvec,
@@ -220,13 +200,8 @@ if __name__ == '__main__':
                     point3D_ids = []
                 )
 
-                if os.path.exists(test_file) and image_meta.name in blending_dict:
-                    n_pts = np.isin(image_meta.point3D_ids, new_indices).sum()
-                    blending_dict[image_meta.name][f"{i}_{j}"] = str(n_pts)
-
-
             points_out = {
-                new_indices[idx] : Point3D(
+                new_indices[idx] : Namespace(
                         id=new_indices[idx],
                         xyz= new_xyzs[idx],
                         rgb=new_colors[idx],
@@ -237,12 +212,14 @@ if __name__ == '__main__':
                 for idx in range(len(new_xyzs))
             }
 
-            write_model(cam_intrinsics, images_out, points_out, out_colmap, f".{args.model_type}")
-
-            with open(os.path.join(out_path, "center.txt"), 'w') as f:
-                f.write(' '.join(map(str, (corner_min + corner_max) / 2)))
-            with open(os.path.join(out_path, "extent.txt"), 'w') as f:
-                f.write(' '.join(map(str, corner_max - corner_min)))
+            return {
+                "cameras": cam_intrinsics,
+                "images": images_out,
+                "points3D": points_out,
+                "center": (corner_min + corner_max) / 2,
+                "extent": corner_max - corner_min,
+                "coords": (i, j)
+            }
         else:
             excluded_chunks.append([i, j])
             print("Chunk excluded")
@@ -253,8 +230,4 @@ if __name__ == '__main__':
 
     for i in range(n_width):
         for j in range(n_height):
-            make_chunk(i, j, n_width, n_height)
-
-    if os.path.exists(test_file):
-        with open(f"{args.base_dir}/blending_dict.json", "w") as f:
-            json.dump(blending_dict, f, indent=2)
\ No newline at end of file
+            yield make_chunk(i, j, n_width, n_height)
""".strip())
