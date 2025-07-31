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
    "depth_lp_loss": "lp_loss.item() if lp_loss is not None else 0.0",
    "depth_pearson_loss": "pearson_loss.item() if pearson_loss is not None else 0.0",
    "diffusion_loss": "diffusion_loss.item() if diffusion_loss is not None else 0.0",
    "reg_loss": "reg_loss.item() if reg_loss is not None else 0.0",
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
    "save_cc",
]

#
# Patch Gaussian Splatting to include masks
# Also, fix cx, cy (ignored in gaussian-splatting)
#
# <fix cameras>
def getProjectionMatrixFromOpenCV(w, h, fx, fy, cx, cy, znear, zfar):
    z_sign = 1.0
    P = torch.zeros((4, 4))  # type: ignore
    P[0, 0] = 2.0 * fx / w
    P[1, 1] = 2.0 * fy / h
    P[0, 2] = (2.0 * cx - w) / w
    P[1, 2] = (2.0 * cy - h) / h
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


# Patch loadCam to pass args to Cameras
@import_context.patch_ast_import("utils.camera_utils")
def _(ast_module: ast.Module):
    # Make sure PILtoTorch is imported
    assert any(isinstance(x, ast.ImportFrom) and x.module == "utils.general_utils" and x.names[0].name == "PILtoTorch" for x in ast_module.body), "PILtoTorch not imported in camera_utils"
    load_cam_ast = next((x for x in ast_module.body if isinstance(x, ast.FunctionDef) and x.name == "loadCam"), None)
    assert load_cam_ast is not None, "loadCam not found in camera_utils"
    return_ast = load_cam_ast.body[-1]
    assert isinstance(return_ast, ast.Return), "loadCam does not return camera"
    camera = return_ast.value
    assert isinstance(camera, ast.Call), "loadCam does not return camera"
    camera.keywords.append(ast.keyword(arg="mask", value=ast.parse("PILtoTorch(cam_info.mask, (gt_image.size[1], gt_image.size[0])) if cam_info.mask is not None else None").body[0].value, lineno=0, col_offset=0))  # type: ignore
    camera.keywords.append(ast.keyword(arg="cx", value=ast.parse("cam_info.cx").body[0].value, lineno=0, col_offset=0))  # type: ignore
    camera.keywords.append(ast.keyword(arg="cy", value=ast.parse("cam_info.cy").body[0].value, lineno=0, col_offset=0))  # type: ignore


# Patch Cameras to include masks and cx, cy
@import_context.patch_ast_import("scene.cameras")
def _(ast_module: ast.Module):
    camera_ast = next((x for x in ast_module.body if isinstance(x, ast.ClassDef) and x.name == "Camera"), None)
    assert camera_ast is not None, "Camera not found in cameras"
    # Add mask and cx, cy to Camera.__init__
    init = next((x for x in camera_ast.body if isinstance(x, ast.FunctionDef) and x.name == "__init__"), None)
    assert init is not None, "Camera.__init__ not found"

    # Assert torch in imports
    assert any(isinstance(x, ast.Import) and any(y.name == "torch" for y in x.names) for x in ast_module.body), "torch not imported in cameras"

    # Add import from utils.graphics_utils import fov2focal
    import_ast = ast.ImportFrom(module="utils.graphics_utils", names=[ast.alias(name="fov2focal", asname=None, lineno=0, col_offset=0)], level=0, lineno=0, col_offset=0)
    ast_module.body.insert(0, import_ast)

    # Add getProjectionMatrixFromOpenCV to scene.cameras
    with open(__file__, "r") as f:
        _self_ast = ast.parse(f.read())
        getProjectionMatrixFromOpenCV_ast = next((x for x in _self_ast.body if isinstance(x, ast.FunctionDef) and x.name == "getProjectionMatrixFromOpenCV"), None)
        assert getProjectionMatrixFromOpenCV_ast is not None, f"getProjectionMatrixFromOpenCV not found in {__file__}"
    ast_module.body.append(getProjectionMatrixFromOpenCV_ast)

    # Add missing arguments
    init.args.kwonlyargs.append(ast.arg(arg="mask", annotation=None, lineno=0, col_offset=0))
    init.args.kw_defaults.append(ast.Constant(value=None, lineno=0, col_offset=0))
    init.args.kwonlyargs.append(ast.arg(arg="cx", annotation=None, lineno=0, col_offset=0))
    init.args.kw_defaults.append(ast.Constant(value=None, lineno=0, col_offset=0))
    init.args.kwonlyargs.append(ast.arg(arg="cy", annotation=None, lineno=0, col_offset=0))
    init.args.kw_defaults.append(ast.Constant(value=None, lineno=0, col_offset=0))

    # Store mask
    init.body.insert(0, ast.parse("self.mask = mask").body[0])

    # Finally, we fix the computed projection matrix
    projection_matrix_ast = next((x for x in init.body if isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == "self.projection_matrix"), None)
    assert projection_matrix_ast is not None, "self.projection_matrix not found in Camera.__init__"
    projection_matrix_ast.value = ast.parse("""getProjectionMatrixFromOpenCV(
    self.image_width, 
    self.image_height, 
    fov2focal(FoVx, self.image_width), 
    fov2focal(FoVy, self.image_height), 
    cx if cx is not None else self.image_width / 2.0, 
    cy if cy is not None else self.image_height / 2.0,
    self.znear, 
    self.zfar).transpose(0, 1).cuda()
""").body[0].value  # type: ignore

@import_context.patch_ast_import("scene.dataset_readers")
def _(ast_module: ast.Module):
    # First remove 'from recordclass import recordclass, RecordClass'
    for i, x in enumerate(ast_module.body):
        if isinstance(x, ast.ImportFrom) and x.module == "recordclass" and any(y.name in ("recordclass", "RecordClass") for y in x.names):
            ast_module.body.pop(i)
            break

    camera_info_ast = next((x for x in ast_module.body if isinstance(x, ast.ClassDef) and x.name == "CameraInfo"), None)
    assert camera_info_ast is not None, "CameraInfo not found in dataset_readers"
    # Add typing import
    ast_module.body.insert(0, ast.ImportFrom(module="typing", names=[ast.alias(name="Optional", asname=None, lineno=0, col_offset=0)], level=0, lineno=0, col_offset=0))

    # Add mask and cx, cy attributes to CameraInfo
    camera_info_ast.body.extend([
        ast.AnnAssign(target=ast.Name(id="mask", ctx=ast.Store(), lineno=0, col_offset=0), 
                      annotation=ast.parse("Optional[np.ndarray]").body[0].value,  # type: ignore
                      value=None, simple=1, lineno=0, col_offset=0),
        ast.AnnAssign(target=ast.Name(id="cx", ctx=ast.Store(), lineno=0, col_offset=0), 
                      annotation=ast.Name(id="float", ctx=ast.Load(), lineno=0, col_offset=0),
                      value=None, simple=1, lineno=0, col_offset=0),
        ast.AnnAssign(target=ast.Name(id="cy", ctx=ast.Store(), lineno=0, col_offset=0), 
                      annotation=ast.Name(id="float", ctx=ast.Load(), lineno=0, col_offset=0),
                      value=None, simple=1, lineno=0, col_offset=0)])
# </fix cameras>


# Patch Scene to take scene_info as input
# <patch scene>
@import_context.patch_ast_import("scene")
def _(ast_module: ast.Module):
    scene_ast = next((x for x in ast_module.body if isinstance(x, ast.ClassDef) and x.name == "Scene"), None)
    assert scene_ast is not None, "Scene not found in scene"
    scene_init = next((x for x in scene_ast.body if isinstance(x, ast.FunctionDef) and x.name == "__init__"), None)
    assert scene_init is not None, "Scene.__init__ not found"

    # Add scene_info to Scene.__init__
    scene_init.args.args.insert(1, ast.arg(arg="scene_info", annotation=None, lineno=0, col_offset=0))

    # Remove dataset detection if statement
    detection_if = next(x for x in scene_init.body if isinstance(x, ast.If) and ast.unparse(x.test) == "os.path.exists(os.path.join(args.source_path, 'sparse'))")
    assert detection_if is not None, "Dataset detection if statement not found"
    scene_init.body.remove(detection_if)

    # Remove code saving scene_info and saving input ply
    if_not_iter_ast = next(x for x in scene_init.body if isinstance(x, ast.If) and ast.unparse(x.test) == "not self.loaded_iter")
    assert if_not_iter_ast is not None, "if not self.loaded_iter not found"
    with_statements = [x for x in if_not_iter_ast.body if isinstance(x, ast.With)]
    assert len(with_statements) == 2, "Expected 2 with statements in if not self.loaded_iter"
    for x in with_statements:
        if_not_iter_ast.body.remove(x)
# </patch scene>

# Patch dataset_readers to export blender_create_pcd
# <patch blender_create_pcd>
@import_context.patch_ast_import("scene.dataset_readers")
def _(ast_module: ast.Module):
    readNerfSyntheticInfo_ast = next((x for x in ast_module.body if isinstance(x, ast.FunctionDef) and x.name == "readNerfSyntheticInfo"), None)
    assert readNerfSyntheticInfo_ast is not None, "readNerfSyntheticInfo not found in dataset_readers"
    # Find condition not os.path.exists(ply_path)
    if_not_exists = next(x for x in readNerfSyntheticInfo_ast.body if isinstance(x, ast.If) and ast.unparse(x.test) == "not os.path.exists(ply_path)")
    assert if_not_exists is not None, "if not os.path.exists(ply_path) not found"
    body = copy.deepcopy(if_not_exists.body)

    # Remove storePly call
    store_ply = body[-1]
    assert isinstance(store_ply, ast.Expr) and isinstance(store_ply.value, ast.Call) and store_ply.value.func.id == "storePly", "storePly not found in if not os.path.exists(ply_path)"  # type: ignore
    body.remove(store_ply)

    # Add new function
    new_function = cast(ast.FunctionDef, ast.parse("def blender_create_pcd():\n    pass").body[0])
    new_function.body = body
    # Add `return pcd`
    new_function.body.append(ast.Return(value=ast.Name(id="pcd", ctx=ast.Load(), lineno=0, col_offset=0), lineno=0, col_offset=0))
    ast_module.body.append(new_function)
# </patch blender_create_pcd>



def ast_remove(tree, callback):
    tree = copy.deepcopy(tree)
    def _rem(tree):
        if isinstance(tree, list):
            return [x for x in (_rem(x) for x in tree) if x is not None]
        sentinel = ast.AST()
        class Transformer(ast.NodeTransformer):
            def visit(self, node):
                if callback(node):
                    return sentinel
                out = self.generic_visit(node)
                if isinstance(out, ast.Dict):
                    valid_indices = [
                        out.keys[i] is not sentinel and out.values[i] is not sentinel 
                        for i in range(len(out.keys))]
                    out = ast.Dict(
                        keys=[x for i, x in enumerate(out.keys) if valid_indices[i]],
                        values=[x for i, x in enumerate(out.keys) if valid_indices[i]]
                    )
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
    return _rem(tree)


def ast_remove_names(tree, names):
    def _callback(node):
        return isinstance(node, ast.Name) and node.id in names
    return ast_remove(tree, _callback)


# Patch train to extract the training loop and init
# <patch train>
@import_context.patch_ast_import("train")
def _(ast_module: ast.Module):
    # Remove instructions from prune_floaters
    def _prune_callback(node):
        # Remove plt.figure
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "figure":
            return True
        # Remove plt.imsave
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "imsave":
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "plt":
                return True
        # Remove os.makedirs
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (node.func.attr == "makedirs" and
                isinstance(node.func.value, ast.Name) and node.func.value.id == "os"):
                return True
        return False
    # Replace prune_floaters in ast_module
    for i, x in enumerate(ast_module.body):
        if isinstance(x, ast.FunctionDef) and x.name == "prune_floaters":
            prune_floaters_ast = ast_remove(x, _prune_callback)
            ast_module.body[i] = prune_floaters_ast
            break
    else:
        assert False, "prune_floaters not found in train"

    training_ast = copy.deepcopy(next(x for x in ast_module.body if isinstance(x, ast.FunctionDef) and x.name == "training"))
    # Patch torch.load => torch.load(..., weights_only=False)
    for node in ast.walk(training_ast):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "load":
            node.keywords.append(ast.keyword(arg="weights_only", value=ast.Constant(value=False, kind=None)))

    # We remove the unused code
    training_ast = ast_remove_names(training_ast, train_step_disabled_names)
    # Remove os.makedirs and scene.save
    def _remove_callback(node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if (node.func.attr in ("makedirs", "save") and
                isinstance(node.func.value, ast.Name) and node.func.value.id == "os"):
                return True
            if (node.func.attr == "save" and
                isinstance(node.func.value, ast.Name) and node.func.value.id == "scene"):
                return True
        return False
    training_ast = ast_remove(training_ast, _remove_callback)

    # Now, we extract the train iteration code
    train_loop = training_ast.body[-1]
    assert isinstance(train_loop, ast.For) and train_loop.target.id == "iteration", "Expected for loop in training"  # type: ignore
    train_step = list(train_loop.body)
    # Add return statement to train_step
    train_step.append(ast.Return(value=ast.Dict(
        keys=[ast.Constant(value=name, kind=None) for name in metrics.keys()], 
        values=[ast.parse(value).body[0].value for value in metrics.values()],  # type: ignore
    )))
    # Extract render_pkg = ... index
    render_pkg_idx = next(i for i, x in enumerate(train_step) if isinstance(x, ast.Assign) and x.targets[0].id == "render_pkg")  # type: ignore
    train_step.insert(render_pkg_idx+1, ast.parse("""
if viewpoint_cam.mask is not None:
    mask = viewpoint_cam.mask.cuda()
    for k in ["render", "rend_normal", "surf_normal", "rend_dist"]:
        render_pkg[k] = render_pkg[k] * mask + (1.0 - mask) * render_pkg[k].detach()
""").body[0])

    # Detect global names
    global_names = set([x.arg for x in training_ast.args.args])
    for instruction in training_ast.body[:-1]:
        for node in ast.walk(instruction):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                global_names.add(node.id)
    # This is passed as global in 3dgs-mcmc
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

    #
    # Use this code to debug when integrating new codebase
    #
    ## print("Train iteration: ")
    ## print(ast.unparse(train_iteration))
    ## setup_train = ast_remove_names(training_ast.body[:-1], ["first_iter"])
    ## for instruction in setup_train:
    ##     Transformer().visit(instruction)
    ## print("Setup train:") 
    ## print(ast.unparse(setup_train))
    ## print()
    ## # print("Train step:")
    ## # print(ast.unparse(train_step))
    ## print("Train step sets: ")
    ## print(train_step_transformer._stored_names)


import_context.apply_patch(r"""
diff --git a/utils/loss_utils.py b/utils/loss_utils.py
index fc4dc06..17ec63e 100755
--- a/utils/loss_utils.py
+++ b/utils/loss_utils.py
@@ -25,1 +24,0 @@ import torch.nn as nn
-pearson = PearsonCorrCoef().cuda()
""".strip())
