import importlib
import shlex
import os
import logging
from pathlib import Path
import click
from gettext import gettext as _
from .train import train_command
from .render import render_command
from . import registry
from .utils import setup_logging
from .communication import RemoteProcessMethod, NB_PREFIX
from .datasets import download_dataset
from .evaluate import evaluate


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


main.add_command("nerfbaselines.viewer", "viewer")
