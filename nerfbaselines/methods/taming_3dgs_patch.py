import itertools
import copy
import ast
from typing import cast
from ._patching import Context

import_context = Context()

# This file includes several patches to 3DGS codebase
# 1. Patch Gaussian Splatting Cameras to include masks
# 2. Patch 3DGS to handle cx, cy correctly
# 3. Patch scene.Scene to take scene_info as input
# 4. Extract train_iteration from train.py
# 5. Extract blender_create_pcd


# It should be generic and apply to most 3DGS implementations.
# Here are the parameters which configure the patch:
metrics = {
    "loss": "loss.item()",
    "l1_loss": "Ll1.item()",
    "psnr": "10 * torch.log10(1 / torch.mean((image - gt_image) ** 2)).item()",
    "num_points": "len(gaussians.get_xyz)",
}
train_step_disabled_names = [
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
    "websockets",
]

def _ast_prune_node(tree, callback):
    if isinstance(tree, list):
        return [x for x in (_ast_prune_node(x, callback) for x in tree) if x is not None]
    sentinel = ast.AST()
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


# Patch train to extract the training loop and init
# <patch train>
@import_context.patch_ast_import("train")
def _(ast_module: ast.Module):
    training_ast = copy.deepcopy(next(x for x in ast_module.body if isinstance(x, ast.FunctionDef) and x.name == "training"))
    # We remove the unused code
    def prune_node(node):
        if isinstance(node, ast.Name) and node.id in train_step_disabled_names:
            return True
        # Remove args.benchmark_dir
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "args" and node.attr == "benchmark_dir":
            return True
        return False
    _ast_prune_node(training_ast, prune_node)

    # Now, we extract the train iteration code
    train_loop = next((x for x in training_ast.body if isinstance(x, ast.For) and x.target.id == "iteration"), None)
    assert train_loop is not None, "Could not find training loop"
    train_step = list(train_loop.body)
    # Add return statement to train_step
    train_step.append(ast.Return(value=ast.Dict(
        keys=[ast.Constant(value=name, kind=None, lineno=0, col_offset=0) for name in metrics.keys()], 
        values=[ast.parse(value).body[0].value for value in metrics.values()],  # type: ignore
        lineno=0, col_offset=0), lineno=0, col_offset=0))
    # Extract render_pkg = ... index
    render_pkg_idx = next(i for i, x in enumerate(train_step) if isinstance(x, ast.Assign) and x.targets[0].id == "render_pkg")  # type: ignore
    train_step.insert(render_pkg_idx+1, ast.parse("""
if viewpoint_cam.mask is not None:
    mask = viewpoint_cam.mask.cuda()
    for k in ["render", "rend_normal", "surf_normal", "rend_dist"]:
        render_pkg[k] = render_pkg[k] * mask + (1.0 - mask) * render_pkg[k].detach()
""").body[0])

    # Detect global names
    setup_body = [x for x in training_ast.body if x != train_loop]
    global_names = set([x.arg for x in training_ast.args.args])
    for instruction in setup_body:
        for node in ast.walk(instruction):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                global_names.add(node.id)
    # Replace all global names with `self`
    class Transformer(ast.NodeTransformer):
        def __init__(self):
            self._stored_names = set()
        def visit_Name(self, node):
            if node.id in global_names:
                if isinstance(node.ctx, ast.Store):
                    self._stored_names.add(node.id)
                return ast.copy_location(
                    ast.Attribute(value=ast.Name(id="self", ctx=ast.Load(), lineno=0, col_offset=0), attr="_"+node.id, ctx=node.ctx), node)
            return self.generic_visit(node)

    train_step_transformer = Transformer()
    for instruction in train_step:
        train_step_transformer.visit(instruction)

    # Define function train_iteration
    train_iteration = cast(ast.FunctionDef, ast.parse("def train_iteration(self, iteration):\n    pass").body[0])
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
            targets=[ast.Name(id=name, ctx=ast.Store(), lineno=0, col_offset=0)],
            value=ast.Attribute(
                value=ast.Name(id="self", ctx=ast.Load(), lineno=0, col_offset=0), 
                attr="_"+name, 
                ctx=ast.Load(lineno=0, col_offset=0),
                lineno=0, col_offset=0), lineno=0, col_offset=0))
    # Drop last two instructions which save state and print training finished
    setup_train = setup_train_start0 + setup_train_start + setup_train[:-2]
    setup_train_function = cast(ast.FunctionDef, ast.parse("def setup_train(self, args, Scene):\n    pass").body[0])
    setup_train_function.body = setup_train
    ast_module.body.append(setup_train_function)

    ## # Use this code to debug when integrating new codebase
    ## print("===== Setup test =====") 
    ## print(ast.unparse(setup_test_function.body))
    ## print()
    ## print("===== Setup train =====") 
    ## print(ast.unparse(setup_train))
    ## print()
    ## print("===== Train step =====")
    ## print(ast.unparse(train_step))
    ## print()
    ## print("===== Train step sets =====")
    ## print(train_step_transformer._stored_names)
    ## print()
    ## print("===== Get argparser =====")
    ## print(ast.unparse(get_argparser))


PATCH = """
diff --git a/scene/__init__.py b/scene/__init__.py
index 2b31398..c31679a 100644
--- a/scene/__init__.py
+++ b/scene/__init__.py
@@ -25 +25 @@ class Scene:
-    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
+    def __init__(self, args : ModelParams, scene_info, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
@@ -43,8 +42,0 @@ class Scene:
-        if os.path.exists(os.path.join(args.source_path, "sparse")):
-            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
-        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
-            print("Found transforms_train.json file, assuming Blender data set!")
-            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
-        else:
-            assert False, "Could not recognize scene type!"
-
@@ -52,2 +43,0 @@ class Scene:
-            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
-                dest_file.write(src_file.read())
@@ -62,2 +51,0 @@ class Scene:
-            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
-                json.dump(json_cams, file)
diff --git a/scene/cameras.py b/scene/cameras.py
index abf6e52..4ef519f 100644
--- a/scene/cameras.py
+++ b/scene/cameras.py
@@ -16 +16,2 @@ from utils.graphics_utils import getWorld2View2, getProjectionMatrix
-
+from utils.graphics_utils import getProjectionMatrixFromOpenCV, fov2focal
+ 
@@ -19 +20 @@ class Camera(nn.Module):
-                 image_name, uid,
+                 image_name, uid, mask, cx, cy,
@@ -30,0 +32 @@ class Camera(nn.Module):
+        self.mask = mask
@@ -55 +57,9 @@ class Camera(nn.Module):
-        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
+        self.projection_matrix = getProjectionMatrixFromOpenCV(
+            self.image_width, 
+            self.image_height, 
+            fov2focal(FoVx, self.image_width), 
+            fov2focal(FoVy, self.image_height), 
+            cx, 
+            cy, 
+            self.znear, 
+            self.zfar).transpose(0, 1).cuda()
diff --git a/scene/dataset_readers.py b/scene/dataset_readers.py
index 2a6f904..3597aee 100644
--- a/scene/dataset_readers.py
+++ b/scene/dataset_readers.py
@@ -15,0 +16 @@ from typing import NamedTuple
+from typing import Optional
@@ -36,0 +38,3 @@ class CameraInfo(NamedTuple):
+    mask: Optional[np.array]
+    cx: float
+    cy: float
@@ -256,0 +261,11 @@ def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
+def blender_create_pcd():
+    # Since this data set has no colmap data, we start with random points
+    num_pts = 100_000
+    print(f"Generating random point cloud ({num_pts})...")
+    
+    # We create random points inside the bounds of the synthetic Blender scenes
+    xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
+    shs = np.random.random((num_pts, 3)) / 255.0
+    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
+    return pcd
+
diff --git a/utils/camera_utils.py b/utils/camera_utils.py
index 1a54d0a..87075d6 100644
--- a/utils/camera_utils.py
+++ b/utils/camera_utils.py
@@ -50,0 +53 @@ def loadCam(args, id, cam_info, resolution_scale):
+                  cx=cam_info.cx, cy=cam_info.cy,
@@ -51,0 +55,2 @@ def loadCam(args, id, cam_info, resolution_scale):
+                  mask=(PILtoTorch(cam_info.mask, resolution) 
+                                 if cam_info.mask is not None else None),
diff --git a/utils/graphics_utils.py b/utils/graphics_utils.py
index b4627d8..33ba2dc 100644
--- a/utils/graphics_utils.py
+++ b/utils/graphics_utils.py
@@ -72,0 +73,12 @@ def getProjectionMatrix(znear, zfar, fovX, fovY):
+def getProjectionMatrixFromOpenCV(w, h, fx, fy, cx, cy, znear, zfar):
+    z_sign = 1.0
+    P = torch.zeros((4, 4))  # type: ignore
+    P[0, 0] = 2.0 * fx / w
+    P[1, 1] = 2.0 * fy / h
+    P[0, 2] = (2.0 * cx - w) / w
+    P[1, 2] = (2.0 * cy - h) / h
+    P[3, 2] = z_sign
+    P[2, 2] = z_sign * zfar / (zfar - znear)
+    P[2, 3] = -(zfar * znear) / (zfar - znear)
+    return P
+
"""
import_context.apply_patch(PATCH)
