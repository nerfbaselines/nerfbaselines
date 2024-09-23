import logging
import click
import nerfbaselines
from nerfbaselines import backends
from ._common import NerfBaselinesCliCommand, click_backend_option


@click.command("shell", context_settings=dict(
    ignore_unknown_options=True,
    allow_interspersed_args=False,
), cls=NerfBaselinesCliCommand, short_help="Run a shell command in the backend environment", help=(
    "Run a shell command in the backend environment. "
    "This command is useful for debugging and running custom commands in the backend environment."
))
@click.option("--method", type=click.Choice(list(nerfbaselines.get_supported_methods())), required=True, help="Method to use.")
@click_backend_option()
@click.argument('command', nargs=-1, type=click.UNPROCESSED)
def shell_command(method, backend_name, command):
    method_spec = nerfbaselines.get_method_spec(method)
    backend_impl = backends.get_backend(method_spec, backend_name)
    logging.info(f"Using method: {method}, backend: {backend_impl.name}")
    backend_impl.install()
    backend_impl.shell(command if command else None)


