# Exporting meshes

For some methods (where applicable), NerfBaselines allows you to export the reconstructed scene as a mesh. Currently, these include `colmap`, `gaussian-opacity-fields`, `2d-gaussian-splatting`. In this tutorial, we will show you how to export a mesh for a trained model. We assume you trained `gaussian-opacity-fields` model on the `mipnerf360/bicycle` scene and have the checkpoint stored in `checkpoint-30000` directory. You can obtain this model by running the following command:

```bash
nerfbaselines train \
    --method gaussian-opacity-fields \
    --data external://mipnerf360/bicycle
```

## Exporting a mesh
To export the web demo, you can use the `nerfbaselines export-mesh` command. The command takes as its input the path to the checkpoint directory, the output directory where the mesh will be stored, and the `--data` argument (required for some methods), which points to the original dataset used to train the methods. In our case, we will use the `--data` argument to specify the `mipnerf360/bicycle` scene.

```bash
nerfbaselines export-mesh \
    --method gaussian-opacity-fields \
    --data external://mipnerf360/bicycle \
    --checkpoint checkpoint-30000 \
    --output mesh
```

This command will export the mesh to the `mesh` directory. In this case, the mesh will be stored in the `mesh/mesh.ply` file. However, some methods may output multiple files or different file formats. You can find more information about the output in the method-specific documentation.
