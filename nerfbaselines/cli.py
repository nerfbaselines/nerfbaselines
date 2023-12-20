import subprocess
import tempfile
import importlib
import shlex
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
from .communication import RemoteProcessMethod, NB_PREFIX
from .datasets import download_dataset
from .evaluate import evaluate
from .results import MethodLink


class LazyGroup(click.Group):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._commands = dict()

    def get_command(self, ctx, cmd_name):
        package = self._commands.get(cmd_name, None)
        if package is not None:
            if isinstance(package, str):
                package = importlib.import_module(package, __name__).main
            return package
        return None

    def list_commands(self, ctx):
        return list(self._commands.keys())

    def add_command(self, package_name, command_name=None):
        if command_name is not None:
            self._commands[command_name] = package_name
        else:
            self._commands[package_name.name] = package_name

    def format_commands(self, ctx, formatter) -> None:
        """Extra format methods for multi methods that adds all the commands
        after the options.
        """
        # allow for 3 times the default spacing
        commands = sorted(self._commands.keys())
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
@click.option("--method", type=click.Choice(list(registry.supported_methods())), required=True)
@click.option("--backend", type=click.Choice(registry.ALL_BACKENDS), default=os.environ.get("NB_BACKEND", None))
@click.option("--verbose", "-v", is_flag=True)
def shell_command(method, backend, verbose):
    logging.basicConfig(level=logging.INFO)
    setup_logging(verbose)

    method_spec = registry.get(method)
    _method, backend = method_spec.build(backend=backend)
    logging.info(f"Using method: {method}, backend: {backend}")

    assert issubclass(_method, RemoteProcessMethod)
    methodobj = _method()
    if hasattr(methodobj, "install"):
        methodobj.install()
    env = methodobj._get_isolated_env()
    env["_NB_IS_DOCKERFILE"] = "1"
    args = methodobj._get_server_process_args(env)
    os.execv("/bin/bash", ["/bin/bash", "-c", shlex.join(args)])


@main.command("download-dataset")
@click.argument("dataset", type=str, required=True)
@click.option("--output", "-o", type=click.Path(file_okay=False, dir_okay=True, path_type=Path), required=False, default=None)
@click.option("--verbose", "-v", is_flag=True)
def download_dataset_command(dataset, output, verbose):
    logging.basicConfig(level=logging.INFO)
    setup_logging(verbose)
    if output is None:
        output = Path(NB_PREFIX) / "datasets" / dataset
    download_dataset(dataset, output)


@main.command("evaluate")
@click.argument("predictions", type=click.Path(file_okay=True, dir_okay=True, path_type=Path), required=True)
@click.option("--output", "-o", type=click.Path(file_okay=True, dir_okay=False, path_type=Path), required=True)
@click.option("--disable-extra-metrics", help="Disable extra metrics which need additional dependencies.", is_flag=True)
def evaluate_command(predictions, output, disable_extra_metrics):
    evaluate(predictions, output, disable_extra_metrics=disable_extra_metrics)


@main.command("render-dataset-results")
@click.option("--results", type=click.Path(file_okay=True, exists=True, dir_okay=True, path_type=Path), required=False)
@click.option("--dataset", type=str, required=False)
@click.option("--output-type", type=click.Choice(["markdown", "json"]), default="markdown")
@click.option("--method-links", type=click.Choice(MethodLink.__args__), default="none")
@click.option("--output", type=click.Path(file_okay=True, exists=False, dir_okay=False, path_type=Path), default=None)
def render_dataset_results_command(results: Path, dataset, output_type, output, method_links="none"):
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
            with output.open("w", encoding="utf8") as f:
                print(output_str, end="", file=f)

    setup_logging(False)
    if results is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.check_call("git clone https://github.com/jkulhanek/nerfbaselines.git --branch results --single-branch".split() + [tmpdir])
            if dataset is None:
                logging.fatal("--dataset must be provided")
            dataset_info = compile_dataset_results(Path(tmpdir) / "results", dataset)
            render_output(dataset_info)
    elif results.is_dir():
        if dataset is None:
            logging.fatal("If --results is a directory, --dataset must be provided")
        dataset_info = compile_dataset_results(results, dataset)
        render_output(dataset_info)
    else:
        with results.open("r", encoding="utf8") as f:
            dataset_info = json.load(f)
        render_output(dataset_info)


main.add_command("nerfbaselines.viewer", "viewer")
