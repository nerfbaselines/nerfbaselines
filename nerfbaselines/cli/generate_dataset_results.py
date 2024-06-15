import click
import subprocess
import tempfile
import os
import logging
from pathlib import Path
import click
import json
from gettext import gettext as _
from nerfbaselines.results import MethodLink
from nerfbaselines.types import Optional, get_args


@click.command("generate-dataset-results")
@click.option("--results", type=click.Path(file_okay=True, exists=True, dir_okay=True, path_type=str), required=False)
@click.option("--dataset", type=str, required=False)
@click.option("--output-type", type=click.Choice(["markdown", "json"]), default="markdown")
@click.option("--method-links", type=click.Choice(get_args(MethodLink)), default="none")
@click.option("--output", type=click.Path(file_okay=True, exists=False, dir_okay=False, path_type=str), default=None)
def main(results: Optional[str], dataset: str, output_type="markdown", output: Optional[str] = None, method_links: MethodLink = "none"):
    from nerfbaselines.results import compile_dataset_results, render_markdown_dataset_results_table
    from nerfbaselines.utils import setup_logging

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
