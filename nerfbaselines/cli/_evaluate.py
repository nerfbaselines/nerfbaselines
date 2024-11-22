import warnings
import click
from nerfbaselines.evaluation import evaluate, run_inside_eval_container, build_evaluation_protocol
from ._common import NerfBaselinesCliCommand
from .._registry import evaluation_protocols_registry


@click.command("evaluate", cls=NerfBaselinesCliCommand, short_help="Evaluate predictions", help=(
    "Evaluate predictions (e.g., obtained by running `nerfbaselines render`) against the ground truth. "
    "The predictions are evaluated using the correct evaluation protocol and the results are saved in the specified JSON output file. "
    "The predictions can be provided as a directory or a tar.gz/zip file."))
@click.argument("predictions", type=click.Path(file_okay=True, dir_okay=True, path_type=str), required=True)
@click.option("--output", "-o", type=click.Path(file_okay=True, dir_okay=False, path_type=str), required=True, help="Path to the output JSON file to save the evaluation results.")
@click.option("--evaluation-protocol", default=None, type=click.Choice(list(evaluation_protocols_registry.keys())), help="Override the default evaluation protocol. WARNING: This is strongly discouraged.", hidden=True)
def evaluate_command(predictions: str, output: str, evaluation_protocol=None) -> None:
    evaluation_protocol_obj = None
    if evaluation_protocol is not None:
        warnings.warn(f"Overriding the evaluation protocol to {evaluation_protocol}. This is strongly discouraged.")
        evaluation_protocol_obj = build_evaluation_protocol(evaluation_protocol)

    with run_inside_eval_container():
        evaluate(predictions, output, evaluation_protocol=evaluation_protocol_obj)


