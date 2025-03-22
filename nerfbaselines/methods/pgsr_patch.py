import itertools
import copy
import ast
from typing import cast
from ._patching import Context

import_context = Context()

metrics = {
    "loss": "loss.item()",
    "l1_loss": "Ll1.item()",
    "ssim": "(1 - Lssim.item()) if 'Lssim' in locals() else (1 - (loss.item() - (1.0 - self._opt.lambda_dssim) * Ll1.item()) / self._opt.lambda_dssim)",
    "psnr": "10 * torch.log10(torch.mean(locals().get('alpha_mask', torch.tensor(1.0))) / torch.mean((image - gt_image) ** 2 * locals().get('alpha_mask', 1))).item()",
    "num_points": "len(gaussians.get_xyz)",
}
train_step_disabled_names = [
    "iter_start", 
    "iter_end",
    "training_report",
    "network_gui", 
    "checkpoint_iterations",
    "saving_iterations",
    "loss_dict",
    "tb_writer",
    "progress_bar",
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

#
# Patch Gaussian Splatting to include sampling masks
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
    load_cam_ast = next((x for x in ast_module.body if isinstance(x, ast.FunctionDef) and x.name == "loadCam"), None)
    assert load_cam_ast is not None, "loadCam not found in camera_utils"
    return_ast = load_cam_ast.body[-1]
    assert isinstance(return_ast, ast.Return), "loadCam does not return camera"
    camera = return_ast.value
    assert isinstance(camera, ast.Call), "loadCam does not return camera"
    camera.keywords.append(ast.keyword(arg="sampling_mask", value=ast.parse("cam_info.sampling_mask").body[0].value, lineno=0, col_offset=0))  # type: ignore
    camera.keywords.append(ast.keyword(arg="image", value=ast.parse("cam_info.image").body[0].value, lineno=0, col_offset=0))  # type: ignore
    camera.keywords.append(ast.keyword(arg="cx", value=ast.parse("cam_info.cx").body[0].value, lineno=0, col_offset=0))  # type: ignore
    camera.keywords.append(ast.keyword(arg="cy", value=ast.parse("cam_info.cy").body[0].value, lineno=0, col_offset=0))  # type: ignore


def process_loaded_image_and_mask(image, sampling_mask, resolution, ncc_scale):
    from PIL import Image
    resized_image_rgb = PILtoTorch(Image.fromarray(image), resolution)
    loaded_mask = None
    if sampling_mask is not None:
        loaded_mask = PILtoTorch(Image.fromarray(sampling_mask), resolution)
    gt_image = resized_image_rgb
    if ncc_scale != 1.0:
        ncc_resolution = (int(resolution[0]/ncc_scale), int(resolution[1]/ncc_scale))
        resized_image_rgb = PILtoTorch(image, ncc_resolution)
    gray_image = (0.299 * resized_image_rgb[0] + 0.587 * resized_image_rgb[1] + 0.114 * resized_image_rgb[2])[None]
    return gt_image, gray_image, loaded_mask


# Patch Cameras to include sampling masks and cx, cy
@import_context.patch_ast_import("scene.cameras")
def _(ast_module: ast.Module):
    camera_ast = next((x for x in ast_module.body if isinstance(x, ast.ClassDef) and x.name == "Camera"), None)
    assert camera_ast is not None, "Camera not found in cameras"
    # Add sampling_mask and cx, cy to Camera.__init__
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
        process_loaded_image_and_mask_ast = next((x for x in _self_ast.body if isinstance(x, ast.FunctionDef) and x.name == process_loaded_image_and_mask.__name__), None)

    ast_module.body.append(getProjectionMatrixFromOpenCV_ast)
    ast_module.body.append(process_loaded_image_and_mask_ast)

    # Add missing arguments
    init.args.args.insert(1, ast.arg(arg="sampling_mask", annotation=None, lineno=0, col_offset=0))
    init.args.args.insert(2, ast.arg(arg="image", annotation=None, lineno=0, col_offset=0))
    init.args.args.insert(3, ast.arg(arg="cx", annotation=None, lineno=0, col_offset=0))
    init.args.args.insert(4, ast.arg(arg="cy", annotation=None, lineno=0, col_offset=0))

    # Store sampling mask
    init.body.insert(0, ast.parse("self.sampling_mask = sampling_mask").body[0])
    init.body.insert(1, ast.parse("self.image = image").body[0])
    init.body.append(ast.parse("self.Cx = cx").body[0])
    init.body.append(ast.parse("self.Cy = cy").body[0])

    # Finally, we fix the computed projection matrix
    projection_matrix_ast = next((x for x in init.body if isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == "self.projection_matrix"), None)
    assert projection_matrix_ast is not None, "self.projection_matrix not found in Camera.__init__"
    projection_matrix_ast.value = ast.parse("""getProjectionMatrixFromOpenCV(
    self.image_width, 
    self.image_height, 
    fov2focal(FoVx, self.image_width), 
    fov2focal(FoVy, self.image_height), 
    cx, 
    cy, 
    self.znear, 
    self.zfar).transpose(0, 1).cuda()
""").body[0].value  # type: ignore

    # Replace all process_image_and_mask with the one defined in this file
    class Transformer(ast.NodeTransformer):
        def visit_Call(self, node):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "process_image":
                return ast.copy_location(ast.Call(
                    func=ast.Name(id="process_loaded_image_and_mask", ctx=ast.Load(), lineno=0, col_offset=0), 
                    args=[
                        ast.Attribute(value=ast.Name(id="self", ctx=ast.Load(), lineno=0, col_offset=0), attr="image", ctx=ast.Load(), lineno=0, col_offset=0),
                        ast.Attribute(value=ast.Name(id="self", ctx=ast.Load(), lineno=0, col_offset=0), attr="sampling_mask", ctx=ast.Load(), lineno=0, col_offset=0),
                        node.args[1],
                        node.args[2],
                    ],
                    keywords=node.keywords), node)
            return self.generic_visit(node)
    ast_module = Transformer().visit(ast_module)

    ## print("===== Camera.__init__ =====")
    ## print(ast.unparse(ast_module))
    ## print()

@import_context.patch_ast_import("scene.dataset_readers")
def _(ast_module: ast.Module):
    camera_info_ast = next((x for x in ast_module.body if isinstance(x, ast.ClassDef) and x.name == "CameraInfo"), None)
    assert camera_info_ast is not None, "CameraInfo not found in dataset_readers"
    # Add typing import
    ast_module.body.insert(0, ast.ImportFrom(module="typing", names=[ast.alias(name="Optional", asname=None, lineno=0, col_offset=0)], level=0, lineno=0, col_offset=0))

    # Add sampling_mask and cx, cy attributes to CameraInfo
    camera_info_ast.body.extend([
        ast.AnnAssign(target=ast.Name(id="sampling_mask", ctx=ast.Store(), lineno=0, col_offset=0), 
                      annotation=ast.parse("Optional[np.ndarray]").body[0].value,  # type: ignore
                      value=None, simple=1, lineno=0, col_offset=0),
        ast.AnnAssign(target=ast.Name(id="image", ctx=ast.Store(), lineno=0, col_offset=0), 
                      annotation=ast.parse("np.ndarray").body[0].value,  # type: ignore
                      value=None, simple=1, lineno=0, col_offset=0),
        ast.AnnAssign(target=ast.Name(id="cx", ctx=ast.Store(), lineno=0, col_offset=0), 
                      annotation=ast.Name(id="float", ctx=ast.Load(), lineno=0, col_offset=0),
                      value=None, simple=1, lineno=0, col_offset=0),
        ast.AnnAssign(target=ast.Name(id="cy", ctx=ast.Store(), lineno=0, col_offset=0), 
                      annotation=ast.Name(id="float", ctx=ast.Load(), lineno=0, col_offset=0),
                      value=None, simple=1, lineno=0, col_offset=0)])
# </fix cameras>


@import_context.patch_ast_import("gaussian_renderer")
def _(ast_module: ast.Module):
    # Add parameter to return_dict = { ... }, add "accumulation": rendered_alpha
    render_ast = next((x for x in ast_module.body if isinstance(x, ast.FunctionDef) and x.name == "render"), None)
    assert render_ast is not None, "render not found in gaussian_renderer"
    ncount = len(render_ast.body)
    render_ast.body.insert(ncount-1, ast.parse("return_dict['accumulation'] = rendered_alpha").body[0])


@import_context.patch_ast_import("scene")
def _(ast_module: ast.Module):
    # Find Scene class
    Scene_ast = single((x for x in ast_module.body if isinstance(x, ast.ClassDef) and x.name == "Scene"))
    Scene_init_ast = single((x for x in Scene_ast.body if isinstance(x, ast.FunctionDef) and x.name == "__init__"))
    Scene_init_ast_idx = Scene_ast.body.index(Scene_init_ast)
    # Remove 'if not self.loaded_iter:'
    if_instance = single((x for x in Scene_init_ast.body if isinstance(x, ast.If) and isinstance(x.test, ast.UnaryOp) and isinstance(x.test.op, ast.Not) and isinstance(x.test.operand, ast.Attribute) and x.test.operand.attr == "loaded_iter"))
    Scene_init_ast.body.remove(if_instance)
    # Remove 'if os.path.exists(os.path.join(args.source_path, "sparse")):'
    if_instance = single((x for x in Scene_init_ast.body if isinstance(x, ast.If) and isinstance(x.test, ast.Call) and isinstance(x.test.func, ast.Attribute) and x.test.func.attr == "exists"))
    Scene_init_ast.body.remove(if_instance)
    # Add scene_info as argument to Scene
    Scene_init_ast.args.args.insert(1, ast.arg(arg="scene_info", annotation=None, lineno=0, col_offset=0))
    # Remove all with statements 
    Scene_init_ast = _ast_prune_node(Scene_init_ast, lambda node: isinstance(node, ast.With))
    Scene_ast.body[Scene_init_ast_idx] = Scene_init_ast
    ## print("===== Scene.__init__ =====")
    ## print(ast.unparse(Scene_ast))


# Patch train to extract the training loop and init
# <patch train>
@import_context.patch_ast_import("train")
def _(ast_module: ast.Module):
    training_ast = copy.deepcopy(next(x for x in ast_module.body if isinstance(x, ast.FunctionDef) and x.name == "training"))
    # Patch torch.load => torch.load(..., weights_only=False)
    for node in ast.walk(training_ast):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "load":
            node.keywords.append(ast.keyword(arg="weights_only", value=ast.Constant(value=False, kind=None, lineno=0, col_offset=0), lineno=0, col_offset=0))
    # We remove the unused code
    training_ast = ast_remove_names(training_ast, train_step_disabled_names)
    # Now, we extract the train iteration code
    train_loop = training_ast.body[-3]
    assert isinstance(train_loop, ast.For) and train_loop.target.id == "iteration", "Expected for loop in training"  # type: ignore
    train_step = list(train_loop.body)
    # Add return statement to train_step
    train_step.append(ast.Return(value=ast.Dict(
        keys=[ast.Constant(value=name, kind=None, lineno=0, col_offset=0) for name in metrics.keys()], 
        values=[ast.parse(value).body[0].value for value in metrics.values()],  # type: ignore
        lineno=0, col_offset=0), lineno=0, col_offset=0))
    # Extract render_pkg = ... index
    render_pkg_idx = next(i for i, x in enumerate(train_step) if isinstance(x, ast.Assign) and getattr(x.targets[0], 'id', None) == "render_pkg")  # type: ignore
    train_step.insert(render_pkg_idx+1, ast.parse("""
if viewpoint_cam.sampling_mask is not None:
    sampling_mask = viewpoint_cam.sampling_mask.cuda()
    for k in ["render", "rend_normal", "surf_normal", "rend_dist"]:
        render_pkg[k] = render_pkg[k] * sampling_mask + (1.0 - sampling_mask) * render_pkg[k].detach()
""").body[0])

    # Detect global names
    global_names = set([x.arg for x in training_ast.args.args])
    for instruction in training_ast.body[:-3]:
        for node in ast.walk(instruction):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                global_names.add(node.id)
    # This is passed as global in 3dgs-mcmc
    global_names.add("args")
    global_names.remove("geo_loss")
    global_names.remove("ncc_loss")
    global_names.remove("normal_loss")
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
        def visit_Call(self, node):
            # Replace AppModel() with AppModel(scene_info)
            if isinstance(node.func, ast.Name) and node.func.id == "AppModel":
                node.args = [ast.parse("max(1600, self._args.num_images)").body[0].value]
            return self.generic_visit(node)

    train_step_transformer = Transformer()
    for instruction in train_step:
        train_step_transformer.visit(instruction)
    train_step.insert(0, ast.parse("normal_loss, geo_loss, ncc_loss = None, None, None").body[0])

    # Define function train_iteration
    train_iteration = cast(ast.FunctionDef, ast.parse("def train_iteration(self, iteration):\n    pass").body[0])
    train_iteration.body = train_step
    # Add `return metrics` where metrics is obtained as {name: eval(name) for name in metrics}
    ast_module.body.append(train_iteration)
    
    #
    # <get_argparser>
    # Now, we setup get_argparser function
    #
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
    # </get_argparser>

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
    # Remove os.system calls
    setup_train = [x for x in setup_train if not isinstance(x, ast.Expr) or not isinstance(x.value, ast.Call) or not isinstance(x.value.func, ast.Attribute) or x.value.func.attr != "system"]
    setup_train = ast_remove_names(setup_train, ["network_gui", "tqdm", "prepare_output_and_logger"])[:-4]
    setup_train_function = cast(ast.FunctionDef, ast.parse("def setup_train(self, args, Scene):\n    pass").body[0])
    setup_train_function.body = setup_train
    ast_module.body.append(setup_train_function)

    # Use this code to debug when integrating new codebase
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
