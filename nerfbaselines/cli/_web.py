import click
from nerfbaselines.web import start_dev_server, build
from ._common import NerfBaselinesCliCommand


web_click_group = click.Group("web")


@web_click_group.command("dev", cls=NerfBaselinesCliCommand, help=(
    "Start a development server for the web benchmark. "
    "Any changes to the source code will be automatically reloaded."
))
@click.option("--data", "data_path", default=None, help="Path to data directory. If not provided, data is generated from the NerfBaselines repository.")
@click.option("--datasets", default=None, help="List of comma separated dataset ids to include.")
@click.option("--port", type=int, default=5500, help="HTTP port for the server.")
@click.option("--docs", "include_docs",
              type=click.Choice(['all', 'latest', 'none']), 
              default="none", 
              show_default=True,
              help="Whether to include the documentation page for all versions, the latest, or none.")
def _(data_path, datasets, include_docs=None, port=5500):
    if include_docs == "none":
        include_docs = None
    datasets = datasets.split(",") if datasets else None
    if include_docs == "none":
        include_docs = None
    start_dev_server(data=data_path, datasets=datasets, include_docs=include_docs, port=port)


@web_click_group.command("build", cls=NerfBaselinesCliCommand, help=(
    "Build the web benchmark static website."
))
@click.option("--data", "data_path", default=None, help="Path to data directory. If not provided, data is generated from the NerfBaselines repository.")
@click.option("--output", required=True, help="Output directory.")
@click.option("--datasets", default=None, help="List of comma separated dataset ids to include.")
@click.option("--base-path", default="", help="Base path for the website.")
@click.option("--docs", "include_docs",
              type=click.Choice(['all', 'latest', 'none']), 
              default="none", 
              show_default=True,
              help="Whether to include the documentation page for all versions, the latest, or none.")
def _(output, data_path, datasets, base_path, include_docs=None):
    if include_docs == "none":
        include_docs = None
    datasets = datasets.split(",") if datasets else None
    build(output, data=data_path, datasets=datasets, include_docs=include_docs, base_path=base_path)
