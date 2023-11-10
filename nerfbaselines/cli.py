import sys
import os
import logging
import click
from .train import train_command
from .render import render_command
from . import registry
from .utils import setup_logging
from .communication import RemoteProcessMethod


@click.group()
def main():
    pass

main.add_command(train_command)
main.add_command(render_command)

@main.command("shell")
@click.option("--method", type=click.Choice(registry.supported_methods()), required=True)
@click.option("--backend", type=click.Choice(registry.ALL_BACKENDS), default=os.environ.get("NB_DEFAULT_BACKEND", None))
@click.option("--verbose", "-v", is_flag=True)
def shell(method, backend, verbose):
    logging.basicConfig(level=logging.INFO)
    setup_logging(verbose)

    method_spec = registry.get(method)
    _method, backend = method_spec.build(backend=backend)
    _method.method_name = method
    logging.info(f"Using method: {_method.method_name}, backend: {backend}")

    assert issubclass(_method, RemoteProcessMethod)
    method = _method()
    method.install()
    env = method._get_isolated_env()
    env["_NB_IS_DOCKERFILE"] = "1"
    args = method._get_server_process_args(env)
    if args[0] == "bash":
        args[0] = "/bin/bash"
    os.execv(args[0], args)