# Custom web benchmark

If you want to host your own benchmark, NerfBaselines allows you to create your own instance of the web, similar to [nerfbaselines.github.io](https://nerfbaselines.github.io). This tutorial will guide you through the process of setting up your own web benchmark.

## Prerequisites
In this tutorial, we will develop a custom instance of the web benchmark. In order to do so, you need to install the additional dependencies needed for the local web development. You can install them by running the following command:

```bash
pip install 'nerfbaselines[web]'
```

## Preparing the data
To create a custom web benchmark, you need to prepare the data to present on the web. The data consists of method output artifacts and the directory structure should look like this:

```
gaussian-opacity-fields/
├── dataset-1/
│   ├── scene-1.zip
│   ├── scene-1.json
│   ├── scene-2.zip
│   ├── scene-2.json
│   └── ...
├── dataset-2/
│   └── ...
└── ...
```

The `zip` files are the output artifacts (checkpoints, predictions, results, tensorboard logs) for each trained method on each scene in the dataset. The `json` files are the results files (`results.json`) stored in the zip files. You can extract these files from the checkpoints manually, or you can use the `scripts/export.sh` script to export the results for all methods and scenes in the dataset.

## Running a local dev server
To run a local development server, you can use the `nerfbaselines web dev` command. The command takes as its input the path to the data directory and the output directory where the web will be stored. In our case, we will use the `--data` argument to specify the directory where all methods' output artifacts are stored (e.g., the parent directory of the `gaussian-opacity-fields` directory). The command optionally also takes `--docs latest|all` to generate the documentation for the latest version of the project or all versions of the project.

```bash
nerfbaselines web dev \
    --data /path/to/data \
    --docs none
```

This command will start a local development server on `http://localhost:5500`. You can access the web by opening this URL in your browser.

## Building the web
After confirming that the web works as expected, you can build the web using the `nerfbaselines web build` command. The command takes the same arguments as the `dev` command, but it also takes the `--output` argument, which specifies the output directory where the web will be stored. Furthermore, if the web will be released in a subdirectory (e.g., `nerfbaselines.github.io/custom`), you can use the `--base-url` argument to specify the base URL of the web.

```bash
nerfbaselines web build \
    --data /path/to/data \
    --output /path/to/output \
    --docs latest
```

After running this command, the static web will be built and stored in the specified output directory. You can now host the web on your own server or GitHub Pages.
