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

    def save(self, path):
        # Save the model
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "params.json"), "w") as f:
            json.dump({"hparams": self.hparams, "step": self.step}, f)

        # Here we save the torch model
        torch.save((self.color, self.optimizer.state_dict()), os.path.join(path, "model.pth"))

    @classmethod
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

    def train_iteration(self, step):
        # Perform a single iteration of the training

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

        # After we have updated the parameters, we should increment the step
        self.step = step + 1

        # Return the stats
        return {
            "loss": loss.item(),
        }

    @torch.no_grad()
    def render(self, camera, *, options=None):
        # Render the images
        w, h = camera.image_sizes

        # Here we simply render a single color image
        yield {
            "color": self.color[None, None, :].expand(h, w, 3).detach().cpu().numpy(),
        }
