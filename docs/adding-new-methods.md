# Adding new methods

## Introduction
The main objective of NerfBaselines is to allow for easy integration of various method into the framework.
Therefore, we tried to make it as easy as possible to integrate new methods.
In this guide, we will show you how to add a new method to NerfBaselines.
In order to integrate a new **python**-based method (we don't cover C++ methods in this guide), 
you need to follow these steps:
1) Create a new Python module for the method. This module will contain the method's interface (which will bind to the official method implementation).
2) Add a `spec` file which describes the method and its dependencies.
3) Register the method with NerfBaselines (so it can be discovered and run).
4) Test the method.
5) (Optional) Open a pull request to merge the method to NerfBaselines repository.

```{tip} 
The source code for the tutorial can be found [here](https://github.com/nerfbaselines/nerfbaselines/tree/main/docs/examples/01-simple-method).
```

## Creating a new method module
The first step is to create a new Python module for the method.
Let's start by adding a file `my_method.py`. We now need to implement the {class}`Method <nerfbaselines.Method>` interface.
```python
from nerfbaselines import Method

class MyMethod(Method):
    def __init__(self, *, checkpoint=None, train_dataset=None, config_overrides=None):
        ...

    @classmethod
    def get_method_info(cls):
        ...

    def get_info(self) -> ModelInfo:
        ...

    @torch.no_grad()
    def render(self, camera, *, options=None):
        ...

    def train_iteration(self, step: int) -> Dict[str, float]:
        ...

    def save(self, path):
        ...
```

In this tutorial, we will implement a simple method that optimizes a single color to be rendered.
We will use PyTorch to demonstrate how pytorch-based methods can be integrated.
Let's start by adding the necessary imports and implementing the {class}`__init__ <nerfbaselines.Method.__init__>` method.
The `__init__` method can be called in two ways. Either with `train_dataset` provided (for training) or without it (for inference).
In either case, `checkpoint` can be provided to load the model from the checkpoint.
There is also an optional `config_overrides` parameter which can be used to override the default hyperparameters.
The `__init__` method should initialize the model, optimizer, and load the checkpoint if provided.
```python
import json, os
from nerfbaselines import Method
from nerfbaselines.utils import convert_image_dtype
import torch.nn
import torch.optim
import torch.nn.functional

class MyMethod(Method):
    def __init__(self, *,
                 checkpoint=None,
                 train_dataset=None,
                 config_overrides=None):
        super().__init__()

        # If train_dataset is not None,
        # initialize the method for training
        self.train_dataset = train_dataset
        self.hparams = {
            "initial_color": [1.0, 0.0, 0.0],
        }
        self._loaded_step = None
        self.step = 0
        self.checkpoint = checkpoint
        if config_overrides is not None:
            self.hparams.update(config_overrides)

        # In this example, we just optimize single color to be rendered
        self.color = torch.nn.Parameter(torch.tensor(self.hparams["initial_color"], dtype=torch.float32))
        self.optimizer = torch.optim.Adam([self.color], lr=1e-3)

        if checkpoint is not None:
            # Load the model from the checkpoint
            with open(os.path.join(checkpoint, "params.json"), "r") as f:
                ckpt_meta = json.load(f)
            self.hparams.update(ckpt_meta["hparams"])
            self._loaded_step = self.step = ckpt_meta["step"]

            # We load the ckpt here
            _state, optim_state = torch.load(os.path.join(checkpoint, "model.pth"))
            self.color.data.copy_(_state)
            self.optimizer.load_state_dict(optim_state)
        else:
            assert train_dataset is not None, "train_dataset must be provided for training"
```

Next, we will implement the {meth}`save <nerfbaselines.Method.save>` method which will save the model to the provided path.

```python
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        # Save the model
        with open(os.path.join(path, "params.json"), "w") as f:
            json.dump({"hparams": self.hparams, "step": self.step}, f)

        # Here we save the torch model
        torch.save((self.color, self.optimizer.state_dict()), os.path.join(path, "model.pth"))
```

Next, we will implement the {meth}`get_method_info <nerfbaselines.Method.get_method_info>`, 
and {meth}`get_info <nerfbaselines.Method.get_info>` methods 
which will return the method's information. This information is used by NerfBaselines to determine 
the method's capabilities and requirements.
```python
    def get_method_info(cls):
        return {
            # Method ID is provided by the registry
            "method_id": "",  

            # Supported camera models (e.g., pinhole, opencv, ...)
            "supported_camera_models": frozenset(("pinhole",)),

            # Features required for training (e.g., color, points3D_xyz, ...)
            "required_features": frozenset(("color",)),

            # Declare supported outputs
            "supported_outputs": ("color",),
        }

    def get_info(self):
        return {
            **self.get_method_info(),
            "hparams": self.hparams,
            "loaded_checkpoint": self.checkpoint,
            "loaded_step": self._loaded_step,

            # This number specifies how many iterations 
            # the method should be trained for.
            "num_iterations": 100,
        }
```

Next, we will implement the {meth}`train_iteration <nerfbaselines.Method.train_iteration>` method which will perform a single iteration of the training.
In this example, we will sample a random image from the training dataset and optimize the color to match the image.
For the purpose of the tutorial we will do this by utilizing PyTorch to show how more complicated methods (e.g., PyTorch based)
can be implemented.
```python
    def train_iteration(self, step):
        # Perform a single iteration of the training
        self.step = step

        # Sample a random image
        rand_idx = torch.randint(len(self.train_dataset["images"]), (1,)).cpu().item()
        image = torch.from_numpy(convert_image_dtype(self.train_dataset["images"][rand_idx][:, :, :3], 'float32'))

        # Compute the loss
        w, h = self.train_dataset["cameras"][rand_idx].image_sizes
        pred = self.color[None, None, :].expand(h, w, 3)
        loss = torch.nn.functional.mse_loss(pred, image)

        # Optimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Return the stats
        return {
            "loss": loss.item(),
        }
```

Finally, we implement the {meth}`render <nerfbaselines.Method.render>` method which will render the image using the provided camera.
```python
    @torch.no_grad()
    def render(self, camera, *, options=None):
        # Render the images
        w, h = camera.image_sizes

        # Here we simply render a single color image
        yield {
            "color": self.color[None, None, :].expand(h, w, 3).detach().cpu().numpy(),
        }
```

Now we have successfully implemented a simple method that optimizes a single color to be rendered.
In the next steps, we will show you how to register the method with NerfBaselines and test it.

## Adding a spec file
The next step is to add a spec file which describes the method and its dependencies.
The spec file is a Python file that contains a {func}`register` call which registers the method with NerfBaselines.
Let's create a file `my_method_spec.py` and add the following content:
```python
from nerfbaselines import register

register({
    "method_class": "my_method:MyMethod",
    "conda": {
        "environment_name": "my_method",
        "python_version": "3.11",
        "install_script": """
# Install PyTorch
pip install torch==2.2.0 torchvision==0.17.0 'numpy<2.0.0' \
    --index-url https://download.pytorch.org/whl/cu118
""",
    },
    "id": "my-method",
    "metadata": {},
})
```

While this is all that is required for the spec and will enable the method to be run using all three backends,
much more information can be provided in the spec file.
For example, you can add method's metadata (e.g., authors, paper, etc.), add results from the paper,
add links to public checkpoints, or add **presets** which will allow users to easily run the method 
with predefined hyperparameters or specify different hyperparameters for different datasets.
For more information, see the spec files of the existing methods.

## Registering the method with NerfBaselines
There are multiple ways to register the method with NerfBaselines.
However, for method development and testing, the easiest way is to add method to the environment variable `NERFBASELINES_REGISTER`.
```bash
export NERFBASELINES_REGISTER="$PWD/my_method_spec.py"
```
Now, you can run:
```bash
nerfbaselines train --help
```
You should see `my-method` in the list of available methods.
All commands that accept the `--method` argument will now accept `my-method` as well.

## Testing the method
To verify that the method is implemented correctly, NerfBaselines provides a testing command `nerfbaselines test-method`.
This command will verify various aspects of the method (e.g., training, rendering, etc.) and will report any issues.
In this tutorial, we will test our method on the `blender/lego` dataset.
```bash
nerfbaselines test-method --method my-method --data external://blender/lego
```

The output should be similar to the following:
```
All tests passed:
  ✓ Method backend initialized
  ✓ Method installed
  ✓ Method info loaded
  ✓ Train dataset loaded
  ✓ Test dataset loaded
  ✓ Model initialized
  ✓ Train iteration passes
  ✓ Eval few passes
  ✓ Eval all passes
  ✓ Render works
  ✓ Saving works
  ✓ Loading from checkpoint (without train dataset) passes
  ✓ Resaving method yields same checkpoint
  ✓ Restored model (without train dataset) matches original
  ✓ Loading from checkpoint (with train dataset) passes
  ✓ Restored model (with train dataset) matches original
  ✓ Full training works
  ✓ Checkpoint reproduces results
  ✓ Final evaluation works and matches predictions
  ⚠ Skipping public checkpoint verification - checkpoint not available
  ⚠ Skipping paper results comparison for fast test
```

## Release the method
Once you are satisfied with the method, you can open a **pull request** to merge the method to NerfBaselines repository.
Alternatively, you can only release the method spec in your repository and instruct users to install it
using `nerfbaselines install` command.
The `nerfbaselines install` command is a more permanent way to install the method (as opposed to using the 
`NERFBASELINES_REGISTER` environment variable), and it will copy the method spec to the NerfBaselines installation directory.
However, we **strongly recommend** opening a pull request to merge the method to NerfBaselines repository as 
it will make the method more discoverable and easier to use for other users.
