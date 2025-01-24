import click
import subprocess
import tempfile
import os
import logging
from typing import Optional
from pathlib import Path
import click
import json
from nerfbaselines.results import MethodLink
from nerfbaselines._constants import RESULTS_REPOSITORY
from ._common import NerfBaselinesCliCommand
try:
    from typing import get_args
except ImportError:
    from typing_extensions import get_args


@click.command("generate-dataset-results", cls=NerfBaselinesCliCommand)
@click.option("--results", type=click.Path(file_okay=True, exists=True, dir_okay=True, path_type=str), required=False)
@click.option("--dataset", type=str, required=False)
@click.option("--output-type", type=click.Choice(["markdown", "json", "latex"]), default="markdown")
@click.option("--method-links", type=click.Choice(get_args(MethodLink)), default="none")
@click.option("--output", type=click.Path(file_okay=True, exists=False, dir_okay=False, path_type=str), default=None)
@click.option("--scenes", type=str, default=None, help="Comma-separated list of scenes to include in the results.")
def main(results: Optional[str], dataset: str, output_type="markdown", output: Optional[str] = None, method_links: MethodLink = "none", scenes=None):
    from nerfbaselines.results import compile_dataset_results, render_markdown_dataset_results_table, render_latex_dataset_results_table

    scenes_list = scenes.split(",") if scenes is not None else None

    def render_output(dataset_info):
        output_str = None
        if output_type == "latex":
            assert method_links == "none", "Method links are not supported in LaTeX output"
            output_str = render_latex_dataset_results_table(dataset_info)
        elif output_type == "markdown":
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

    if results is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            subprocess.check_call(f"git clone --quiet --depth=1 https://{RESULTS_REPOSITORY}".split() + [tmpdir], env={"GIT_LFS_SKIP_SMUDGE": "1"})
            if dataset is None:
                logging.fatal("--dataset must be provided")
            dataset_info = compile_dataset_results(Path(tmpdir), dataset, scenes=scenes_list)
            render_output(dataset_info)
    elif os.path.isdir(results):
        if dataset is None:
            logging.fatal("If --results is a directory, --dataset must be provided")
        dataset_info = compile_dataset_results(results, dataset, scenes=scenes_list)
        render_output(dataset_info)
    else:
        with open(results, "r", encoding="utf8") as f:
            dataset_info = json.load(f)
        render_output(dataset_info)
