# Creating web demos

For some methods, NerfBaselines allows you to export a web demo - a standalone HTML website that shows you the trained model and allows you to interact with it in the web browser. Unlike opening the viewer, it doesn't require a local server to run, and can be easily shared with others. Web demos are currently only supported for mesh-based and 3DGS-based methods.
In this tutorial, we will show you how to export a web demo for a trained model. We assume you trained `gaussian-splatting` model on the `mipnerf360/bicycle` scene and have the checkpoint stored in `checkpoint-30000` directory. You can obtain this model by running the following command:

```bash
nerfbaselines train --method gaussian-splatting --data external://mipnerf360/bicycle
```

## Exporting a web demo
To export the web demo, you can use the `nerfbaselines export-demo` command. Simply provide the path to the checkpoint directory and the output directory where the web demo will be stored. However, we also recommend providing the `--data` argument as for some method this is required to properly setup the demo. In our case, we will use the `--data` argument to specify the `mipnerf360/bicycle` scene.

```bash
nerfbaselines export-demo --method gaussian-splatting --data external://mipnerf360/bicycle --checkpoint checkpoint-30000 --output web-demo
```

This command will export the web demo to the `web-demo` directory. You can run a local http server to view the demo by running the following command:

```bash
python -m http.server --directory web-demo 8000
```

This will start a local server on port 8000. You can now open your web browser and navigate to `http://localhost:8000` to view the demo.

## Configuring the web demo
Each method has its own set of parameters that can be configured in the web demo. You can provide these parameters using the `--set` argument, similarly how you provide the arguments for training. In general, there are the following arguments you can consider setting:
- `mock_cors` - if set to `true`, the demo will use a service worker to patch requests which does not have `Access-Control-Allow-Origin` header. This is useful when you want to access the demo file from a different domain without having control over the server.
- `enable_shared_memory` - (*only applicable to 3DGS-based methods*) if set to `true`, the demo will use shared memory to communicate between the worker and the main thread. This can improve the performance of the demo, but it requires support for shared memory in the browser and specific safety headers to be set in the request. These headers can also be patched by setting `mock_cors=true`, but it requires `https` connection..

## Applying apperance embeddings
If you trained the model with appearance embeddings, you can also provide the `--train-embedding {i}` argument to the `export-demo` command, where `i` is smaller then the size of the training dataset. 
This will render the demo with the appearance of the `i-th` training image. For example, to render the demo with the appearance of the 10-th training image, you can run the following command:

```bash
nerfbaselines export-demo --method gaussian-splatting --data external://mipnerf360/bicycle --checkpoint checkpoint-30000 --output web-demo --train-embedding 10
```
