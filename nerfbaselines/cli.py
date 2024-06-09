import subprocess
import itertools
import tempfile
import importlib
import os
import logging
from pathlib import Path
import click
import json
from gettext import gettext as _
from .train import train_command
from .render import render_command
from . import registry
from .utils import setup_logging
from .datasets import download_dataset
from .evaluate import evaluate, run_inside_eval_container
from .results import MethodLink
from .types import Optional, get_args, NB_PREFIX
from .train import Trainer
from . import backends


class LazyGroup(click.Group):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._lazy_commands = dict()

    def get_command(self, ctx, cmd_name):
        package = self._lazy_commands.get(cmd_name, None)
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

    def add_lazy_command(self, package_name: str, command_name: str):
        self._lazy_commands[command_name] = package_name

    def format_commands(self, ctx, formatter) -> None:
        """Extra format methods for multi methods that adds all the commands
        after the options.
        """
        # allow for 3 times the default spacing
        commands = list(sorted(itertools.chain(self._lazy_commands.keys(), self.commands.keys())))
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


main.add_command(train_command)
main.add_command(render_command)


@main.command("shell")
@click.option("--method", type=click.Choice(list(registry.get_supported_methods())), required=True)
@click.option("--backend", type=click.Choice(backends.ALL_BACKENDS), default=os.environ.get("NERFBASELINES_BACKEND", None))
@click.option("--verbose", "-v", is_flag=True)
def shell_command(method, backend, verbose):
    logging.basicConfig(level=logging.INFO)
    setup_logging(verbose)

    method_spec = registry.get(method)
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


@main.command("render-dataset-results")
@click.option("--results", type=click.Path(file_okay=True, exists=True, dir_okay=True, path_type=str), required=False)
@click.option("--dataset", type=str, required=False)
@click.option("--output-type", type=click.Choice(["markdown", "json"]), default="markdown")
@click.option("--method-links", type=click.Choice(get_args(MethodLink)), default="none")
@click.option("--output", type=click.Path(file_okay=True, exists=False, dir_okay=False, path_type=str), default=None)
def render_dataset_results_command(results: Optional[str], dataset: str, output_type="markdown", output: Optional[str] = None, method_links: MethodLink = "none"):
    from .results import compile_dataset_results, render_markdown_dataset_results_table
    from .utils import setup_logging

    def render_output(dataset_info):
        output_str = None
        if output_type == "markdown":
            output_str = render_markdown_dataset_results_table(dataset_info, method_links=method_links)
        elif output_type == "json":
            output_str = json.dumps(dataset_info, indent=2) + os.linesep
        else:
            raise RuntimeError(f"Output type {output_type} is not supported")
        if output is None:
            print(output_str, end="")
        else:
            with open(output, "w", encoding="utf8") as f:
                print(output_str, end="", file=f)

    setup_logging(False)
    if results is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.check_call("git clone https://huggingface.co/jkulhanek/nerfbaselines".split() + [tmpdir], env={"GIT_LFS_SKIP_SMUDGE": "1"})
            if dataset is None:
                logging.fatal("--dataset must be provided")
            dataset_info = compile_dataset_results(Path(tmpdir), dataset)
            render_output(dataset_info)
    elif os.path.isdir(results):
        if dataset is None:
            logging.fatal("If --results is a directory, --dataset must be provided")
        dataset_info = compile_dataset_results(results, dataset)
        render_output(dataset_info)
    else:
        with open(results, "r", encoding="utf8") as f:
            dataset_info = json.load(f)
        render_output(dataset_info)


@main.command("docker-build-image", hidden=True)
@click.option("--method", type=click.Choice(list(registry.get_supported_methods())), required=False)
@click.option("--skip-if-exists-remotely", is_flag=True)
@click.option("--push", is_flag=True)
def build_docker_image_command(method=None, push=False, skip_if_exists_remotely=False):
    from .backends._docker import build_docker_image, get_docker_spec
    spec = registry.get(method) if method is not None else None
    if spec is not None:
        spec = get_docker_spec(spec)
        if spec is None:
            raise RuntimeError(f"Method {method} does not support building docker images")
    build_docker_image(spec, skip_if_exists_remotely=skip_if_exists_remotely, push=push)


main.add_lazy_command("nerfbaselines.export_demo", "export-demo")
main.add_lazy_command("nerfbaselines.viewer", "viewer")
main.add_lazy_command("nerfbaselines.render_trajectory", "render-trajectory")
main.add_lazy_command("nerfbaselines._test_method", "test-method")
main.add_lazy_command("nerfbaselines._generate_web", "generate-web")
main.add_lazy_command("nerfbaselines._fix_checkpoint:fix_checkpoint_command", "fix-checkpoint")
