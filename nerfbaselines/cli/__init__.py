import numpy as np
import sys
import subprocess
import itertools
import tempfile
import importlib
import os
import logging
from pathlib import Path
import click
import json
from typing import Union
from gettext import gettext as _
from nerfbaselines import registry
from nerfbaselines import backends
from nerfbaselines.utils import setup_logging
from nerfbaselines.utils import run_inside_eval_container, handle_cli_error
from nerfbaselines.datasets import download_dataset, load_dataset
from nerfbaselines.types import Optional, get_args, NB_PREFIX, Method
from nerfbaselines.io import load_trajectory, open_any
from nerfbaselines.io import open_any_directory, deserialize_nb_info
from nerfbaselines.evaluation import evaluate, render_all_images, render_frames, trajectory_get_embeddings, trajectory_get_cameras, OutputType


class LazyGroup(click.Group):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._lazy_commands = dict()

    def get_command(self, ctx, cmd_name):
        cmd_def = self._lazy_commands.get(cmd_name, None)
        package = cmd_def.get("command", None) if cmd_def is not None else None
        if package is not None:
            if isinstance(package, str):
                fname = "main"
                if ":" in package:
                    package, fname = package.split(":")
                package = getattr(importlib.import_module(package, __name__), fname)
            return package
        return super().get_command(ctx, cmd_name)

    def list_commands(self, ctx):
        return list(sorted(itertools.chain(self._lazy_commands.keys(), self.commands.keys())))

    def add_lazy_command(self, package_name: str, command_name: str, hidden=False):
        self._lazy_commands[command_name] = dict(
            command=package_name,
            hidden=hidden,
        )

    def format_commands(self, ctx, formatter) -> None:
        """Extra format methods for multi methods that adds all the commands
        after the options.
        """
        # allow for 3 times the default spacing
        commands = list(sorted(itertools.chain((k for k, v in self._lazy_commands.items() if not v["hidden"]), self.commands.keys())))
        if len(commands):
            # limit = formatter.width - 6 - max(len(cmd[0]) for cmd in commands)
            rows = []
            for subcommand in commands:
                rows.append((subcommand, ""))

            with formatter.section(_("Commands")):
                formatter.write_dl(rows)


@click.group(cls=LazyGroup)
def main():
    pass


@main.command("shell")
@click.option("--method", type=click.Choice(list(registry.get_supported_methods())), required=True)
@click.option("--backend", type=click.Choice(backends.ALL_BACKENDS), default=os.environ.get("NERFBASELINES_BACKEND", None))
@click.option("--verbose", "-v", is_flag=True)
def shell_command(method, backend, verbose):
    logging.basicConfig(level=logging.INFO)
    setup_logging(verbose)

    method_spec = registry.get_method_spec(method)
    backend_impl = backends.get_backend(method_spec, backend)
    logging.info(f"Using method: {method}, backend: {backend_impl.name}")
    backend_impl.install()
    backend_impl.shell()


@main.command("download-dataset")
@click.argument("dataset", type=str, required=True)
@click.option("--output", "-o", type=click.Path(file_okay=False, dir_okay=True, path_type=str), required=False, default=None)
@click.option("--verbose", "-v", is_flag=True)
def download_dataset_command(dataset: str, output: str, verbose: bool):
    logging.basicConfig(level=logging.INFO)
    setup_logging(verbose)
    if output is None:
        output = str(Path(NB_PREFIX) / "datasets" / dataset)
    download_dataset(dataset, output)


@main.command("evaluate")
@click.argument("predictions", type=click.Path(file_okay=True, dir_okay=True, path_type=str), required=True)
@click.option("--output", "-o", type=click.Path(file_okay=True, dir_okay=False, path_type=str), required=True)
def evaluate_command(predictions: str, output: str):
    with run_inside_eval_container():
        evaluate(predictions, output)


@main.command("render")
@click.option("--checkpoint", type=str, default=None, required=True)
@click.option("--data", type=str, default=None, required=True)
@click.option("--output", type=str, default="predictions", help="output directory or tar.gz file")
@click.option("--split", type=str, default="test")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--backend", "backend_name", type=click.Choice(backends.ALL_BACKENDS), default=os.environ.get("NERFBASELINES_BACKEND", None))
@handle_cli_error
def render_command(checkpoint: str, data: str, output: str, split: str, verbose: bool, backend_name):
    checkpoint = str(checkpoint)
    setup_logging(verbose)

    # Read method nb-info
    with open_any_directory(checkpoint, mode="r") as _checkpoint_path:
        checkpoint_path = Path(_checkpoint_path)
        assert checkpoint_path.exists(), f"checkpoint path {checkpoint} does not exist"
        assert (checkpoint_path / "nb-info.json").exists(), f"checkpoint path {checkpoint} does not contain nb-info.json"
        with (checkpoint_path / "nb-info.json").open("r") as f:
            nb_info = json.load(f)
        nb_info = deserialize_nb_info(nb_info)

        method_name = nb_info["method"]
        backends.mount(checkpoint_path, checkpoint_path)
        with registry.build_method(method_name, backend=backend_name) as method_cls:
            method: Method = method_cls(checkpoint=str(checkpoint_path))
            method_info = method.get_info()
            dataset = load_dataset(data, 
                                   split=split, 
                                   features=method_info.get("required_features", None), 
                                   supported_camera_models=method_info.get("supported_camera_models", None))
            dataset_colorspace = dataset["metadata"].get("color_space", "srgb")

            if dataset_colorspace != nb_info.get("color_space", "srgb"):
                raise RuntimeError(f"Dataset color space {dataset_colorspace} != method color space {nb_info.get('color_space', 'srgb')}")
            for _ in render_all_images(method, dataset, output=output, nb_info=nb_info):
                pass


@main.command("render-trajectory")
@click.option("--checkpoint", type=str, required=True)
@click.option("--trajectory", type=str, required=True)
@click.option("--output", type=click.Path(path_type=str), default=None, help="output a mp4/directory/tar.gz file")
@click.option("--output-type", type=click.Choice(get_args(OutputType)), default="color", help="output type")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--backend", "backend_name", type=click.Choice(backends.ALL_BACKENDS), default=os.environ.get("NERFBASELINES_BACKEND", None))
@handle_cli_error
def render_trajectory_command(checkpoint: Union[str, Path], trajectory: str, output: Union[str, Path], output_type: OutputType, verbose, backend_name):
    checkpoint = str(checkpoint)
    setup_logging(verbose)

    if os.path.exists(output):
        logging.critical("Output path already exists")
        sys.exit(1)

    # Parse trajectory
    with open_any(trajectory, "r") as f:
        _trajectory = load_trajectory(f)
    cameras = trajectory_get_cameras(_trajectory)

    # Read method nb-info
    logging.info(f"Loading checkpoint {checkpoint}")
    with open_any_directory(checkpoint, mode="r") as _checkpoint_path:
        checkpoint_path = Path(_checkpoint_path)
        assert checkpoint_path.exists(), f"checkpoint path {checkpoint} does not exist"
        assert (checkpoint_path / "nb-info.json").exists(), f"checkpoint path {checkpoint} does not contain nb-info.json"
        with (checkpoint_path / "nb-info.json").open("r") as f:
            nb_info = json.load(f)
        nb_info = deserialize_nb_info(nb_info)

        method_name = nb_info["method"]
        backends.mount(checkpoint_path, checkpoint_path)
        with registry.build_method(method_name, backend=backend_name) as method_cls:
            # dataset, scene = nb_info["dataset_metadata"]["dataset"], nb_info["dataset_metadata"]["scene"]
            # train_dataset = load_dataset(f"external://{dataset}/{scene}", split="train", load_features=False)
            train_dataset = None
            method = method_cls(checkpoint=str(checkpoint_path), train_dataset=train_dataset)

            # Embed the appearance
            embeddings = trajectory_get_embeddings(method, _trajectory)

            render_frames(method, cameras, embeddings=embeddings, output=output, output_type=output_type, nb_info=nb_info, fps=_trajectory["fps"])
            logging.info(f"Output saved to {output}")


@main.command("install-method", hidden=True)
@click.option("--method", type=click.Choice(list(registry.get_supported_methods())), required=False)
@click.option("--backend", "backend_name", type=click.Choice(backends.ALL_BACKENDS), default=os.environ.get("NERFBASELINES_BACKEND", None))
@click.option("--verbose", "-v", is_flag=True)
@handle_cli_error
def install_command(method, backend_name, verbose=False):
    setup_logging(verbose)
    method_spec = registry.get_method_spec(method)
    backend_impl = backends.get_backend(method_spec, backend_name)
    logging.info(f"Using method: {method}, backend: {backend_impl.name}")
    backend_impl.install()


@main.command("build-docker-image", hidden=True)
@click.option("--method", type=click.Choice(list(registry.get_supported_methods("docker"))), required=False)
@click.option("--environment", type=str, required=False)
@click.option("--skip-if-exists-remotely", is_flag=True)
@click.option("--tag-latest", is_flag=True)
@click.option("--push", is_flag=True)
@click.option("--verbose", "-v", is_flag=True)
def build_docker_image_command(method=None, environment=None, push=False, skip_if_exists_remotely=False, tag_latest=False, verbose=False):
    from nerfbaselines.backends._docker import build_docker_image, get_docker_spec
    setup_logging(verbose=verbose)

    spec = None
    if method is not None:
        spec = registry.get_method_spec(method)
        if spec is None:
            raise RuntimeError(f"Method {method} not found")
        spec = get_docker_spec(spec)
        if spec is None:
            raise RuntimeError(f"Method {method} does not support building docker images")
        env_name = spec["environment_name"]
        logging.info(f"Building docker image for environment {env_name} (from method {method})")
    elif environment is not None:
        for method in registry.get_supported_methods("docker"):
            spec = registry.get_method_spec(method)
            spec = get_docker_spec(spec)
            if spec is None:
                continue
            if spec.get("environment_name") == environment:
                break
        if spec is None:
            raise RuntimeError(f"Environment {environment} not found")
        logging.info(f"Building docker image for environment {environment}")
    else:
        logging.info("Building base docker image")
    build_docker_image(spec, skip_if_exists_remotely=skip_if_exists_remotely, push=push, tag_latest=tag_latest)


main.add_lazy_command("nerfbaselines.viewer", "viewer")
main.add_lazy_command("nerfbaselines.cli.export_demo", "export-demo")
main.add_lazy_command("nerfbaselines.cli.test_method", "test-method")
main.add_lazy_command("nerfbaselines.cli.generate_web", "generate-web", hidden=True)
main.add_lazy_command("nerfbaselines.cli.generate_dataset_results:main", "generate-dataset-results")
main.add_lazy_command("nerfbaselines.cli.fix_checkpoint:main", "fix-checkpoint")
main.add_lazy_command("nerfbaselines.cli.fix_output_artifact:main", "fix-output-artifact")
main.add_lazy_command("nerfbaselines.training:train_command", "train")
