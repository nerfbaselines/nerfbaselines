import click
from nerfbaselines.evaluation import evaluate, run_inside_eval_container
from ._common import NerfBaselinesCliCommand


@click.command("evaluate", cls=NerfBaselinesCliCommand, short_help="Evaluate predictions", help=(
    "Evaluate predictions (e.g., obtained by running `nerfbaselines render`) against the ground truth. "
    "The predictions are evaluated using the correct evaluation protocol and the results are saved in the specified JSON output file. "
    "The predictions can be provided as a directory or a tar.gz/zip file."))
@click.argument("predictions", type=click.Path(file_okay=True, dir_okay=True, path_type=str), required=True)
@click.option("--output", "-o", type=click.Path(file_okay=True, dir_okay=False, path_type=str), required=True, help="Path to the output JSON file to save the evaluation results.")
def evaluate_command(predictions: str, output: str):
    with run_inside_eval_container():
        evaluate(predictions, output)


