import numpy as np
import os
import nerfbaselines.registry
from nerfbaselines.types import Method
from nerfbaselines.datasets import load_dataset
from nerfbaselines.io import open_any_directory

import tqdm

frame = {
    "color": np.random.rand(1080, 1920, 3),
    "depth": np.random.rand(1080, 1920),
}



class MeasureFPSMethod:
    def __init__(self, *args, **kwargs):
        pass

    def get_info(self):
        return {}

    @staticmethod
    def get_method_info():
        return {}

    def render(self, cameras, *args, **kwargs):
        for c in cameras:
            yield frame

spec = {
    "id": "test",
    "method": "measure_fps:MeasureFPSMethod",
    "conda": {
        "environment_name": "test",
        "python_version": "3.11",
        "install_script": "",
    }
}



if __name__ == "__main__":
    with nerfbaselines.registry.build_method(spec, backend=os.environ.get("NB_BACKEND")) as method_cls:
        method = method_cls()

        # Warmup
        cameras = [1]
        list(method.render(cameras))
        list(method.render(cameras))
        list(method.render(cameras))
        list(method.render(cameras))
        
        # Measure FPS
        import time
        start_time = time.time()
        nframes = 100
        for _ in tqdm.trange(nframes, disable=True):
            list(method.render(cameras))
        elapsed_time = time.time() - start_time
        print(f"FPS: {nframes / elapsed_time:.2f}")
