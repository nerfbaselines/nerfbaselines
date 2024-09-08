import itertools
import importlib
import logging
from pathlib import Path
import click
import nerfbaselines
from nerfbaselines import backends, NB_PREFIX
from nerfbaselines.datasets import download_dataset
from nerfbaselines.evaluation import evaluate, run_inside_eval_container
from ._web import web_click_group
from ._common import click_backend_option as _click_backend_option
from ._common import setup_logging


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
        del ctx
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

            with formatter.section("Commands"):
                formatter.write_dl(rows)


@click.group(name="nerfbaselines", cls=LazyGroup)
def main():
    pass


@main.command("shell", context_settings=dict(
    ignore_unknown_options=True,
    allow_interspersed_args=False,
))
@click.option("--method", type=click.Choice(list(nerfbaselines.get_supported_methods())), required=True)
@click.option("--verbose", "-v", is_flag=True)
@_click_backend_option()
@click.argument('command', nargs=-1, type=click.UNPROCESSED)
def shell_command(method, backend_name, verbose, command):
    logging.basicConfig(level=logging.INFO)
    setup_logging(verbose)

    method_spec = nerfbaselines.get_method_spec(method)
    backend_impl = backends.get_backend(method_spec, backend_name)
    logging.info(f"Using method: {method}, backend: {backend_impl.name}")
    backend_impl.install()
    backend_impl.shell(command if command else None)


@main.command("download-dataset")
@click.argument("dataset", type=str, required=True)
@click.option("--output", "-o", type=click.Path(file_okay=False, dir_okay=True, path_type=str), required=False, default=None)
@click.option("--verbose", "-v", is_flag=True)
def download_dataset_command(dataset: str, output: str, verbose: bool):
    logging.basicConfig(level=logging.INFO)
    setup_logging(verbose)
    if output is None:
        _out_dataset = dataset
        if _out_dataset.startswith("external://"):
            _out_dataset = _out_dataset[len("external://") :]
        output = str(Path(NB_PREFIX) / "datasets" / _out_dataset)
    download_dataset(dataset, output)
    logging.info(f"Dataset {dataset} downloaded to {output}")


@main.command("evaluate")
@click.argument("predictions", type=click.Path(file_okay=True, dir_okay=True, path_type=str), required=True)
@click.option("--output", "-o", type=click.Path(file_okay=True, dir_okay=False, path_type=str), required=True)
def evaluate_command(predictions: str, output: str):
    with run_inside_eval_container():
        evaluate(predictions, output)


@main.command("build-docker-image", hidden=True)
@click.option("--method", type=click.Choice(list(nerfbaselines.get_supported_methods("docker"))), required=False)
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
        spec = nerfbaselines.get_method_spec(method)
        if spec is None:
            raise RuntimeError(f"Method {method} not found")
        spec = get_docker_spec(spec)
        if spec is None:
            raise RuntimeError(f"Method {method} does not support building docker images")
        env_name = spec["environment_name"]
        logging.info(f"Building docker image for environment {env_name} (from method {method})")
    elif environment is not None:
        for method in nerfbaselines.get_supported_methods("docker"):
            spec = nerfbaselines.get_method_spec(method)
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


main.add_command(web_click_group)
main.add_lazy_command("nerfbaselines.cli._export_demo", "export-demo")
main.add_lazy_command("nerfbaselines.cli._test_method", "test-method")
main.add_lazy_command("nerfbaselines.cli._render:render_command", "render")
main.add_lazy_command("nerfbaselines.cli._render:render_trajectory_command", "render-trajectory")
main.add_lazy_command("nerfbaselines.cli._generate_dataset_results:main", "generate-dataset-results")
main.add_lazy_command("nerfbaselines.cli._fix_checkpoint:main", "fix-checkpoint")
main.add_lazy_command("nerfbaselines.cli._install_method:install_method_command", "install-method")
main.add_lazy_command("nerfbaselines.cli._fix_output_artifact:main", "fix-output-artifact")
main.add_lazy_command("nerfbaselines.cli._train:train_command", "train")
main.add_lazy_command("nerfbaselines.cli._viewer:viewer_command", "viewer")
