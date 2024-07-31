import sys
import itertools
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
from nerfbaselines.types import get_args, NB_PREFIX, Method
from nerfbaselines.io import load_trajectory, open_any
from nerfbaselines.io import open_any_directory, deserialize_nb_info
from nerfbaselines.evaluation import evaluate, render_all_images, render_frames, trajectory_get_embeddings, trajectory_get_cameras, OutputType
from nerfbaselines.web import get_click_group as get_web_click_group


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
        return sorted(itertools.chain(self._lazy_commands.keys(), self.commands.keys()))

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
        commands = []
        lazy_cmds = ((k, v) for k, v in self._lazy_commands.items() if not v["hidden"])
        for name, cmd in sorted(itertools.chain(lazy_cmds, self.commands.items()), key=lambda x: x[0]):
            if isinstance(cmd, click.Group):
                for cmd2 in cmd.list_commands(ctx):
                    sub_cmd = cmd.get_command(ctx, cmd2)
                    if sub_cmd is not None:
                        commands.append(" ".join((name, cmd2)))
            else:
                commands.append(name)

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


main.add_command(get_web_click_group())
main.add_lazy_command("nerfbaselines.viewer", "viewer")
main.add_lazy_command("nerfbaselines.cli.export_demo", "export-demo")
main.add_lazy_command("nerfbaselines.cli.test_method", "test-method")
main.add_lazy_command("nerfbaselines.cli.render:render_command", "render")
main.add_lazy_command("nerfbaselines.cli.render:render_trajectory_command", "render-trajectory")
main.add_lazy_command("nerfbaselines.cli.generate_web", "generate-web", hidden=True)
main.add_lazy_command("nerfbaselines.cli.generate_dataset_results:main", "generate-dataset-results")
main.add_lazy_command("nerfbaselines.cli.fix_checkpoint:main", "fix-checkpoint")
main.add_lazy_command("nerfbaselines.cli.install_method:main", "install-method")
main.add_lazy_command("nerfbaselines.cli.fix_output_artifact:main", "fix-output-artifact")
main.add_lazy_command("nerfbaselines.training:train_command", "train")
