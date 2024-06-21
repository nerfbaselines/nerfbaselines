import os
import sys
from pathlib import Path
import click

try:
    import nerfbaselines  # noqa
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    import nerfbaselines  # noqa
from nerfbaselines.cli.generate_dataset_results import main as generate_dataset_results_command
from nerfbaselines.results import get_dataset_info
from nerfbaselines._constants import WEBPAGE_URL
import tempfile


def update_dataset_results(readme: str, dataset):
    lines = readme.rstrip(os.linesep).splitlines()

    def simplify(s: str):
        return s.lower().replace(" ", "").replace("-", "").replace("_", "")

    # Locate old section
    section_start = next((x for x in range(len(lines)) if simplify(f"### {dataset}") in simplify(lines[x])), None)
    if section_start is None:
        click.echo(click.style(f"Could not locate dataset {dataset} in README.md", fg="bright_yellow"))
        return readme
    section_end = next((x for x in range(section_start + 1, len(lines)) if lines[x].startswith("##")), len(lines))

    # def generate_dataset_results_command(results: Path, dataset, output_type, output, method_links="none"):
    assert generate_dataset_results_command.callback is not None
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_dataset_results_command.callback(None, dataset, output_type="markdown", output=Path(os.path.join(tmpdir, "results.md")), method_links="results")
        new_results = (Path(tmpdir) / "results.md").read_text()

    # Replace old results with new results
    dataset_info = get_dataset_info(dataset)
    new_section = f"""{lines[section_start]}
{dataset_info['description']}
Detailed results are available on the project page: [{WEBPAGE_URL}/{dataset}]({WEBPAGE_URL}/{dataset})

{new_results}
"""
    return os.linesep.join(lines[:section_start] + [new_section] + lines[section_end:]) + os.linesep


@click.command("update-readme")
def main():
    # def generate_dataset_results_command(results: Path, dataset, output_type, output, method_links="none"):
    readme_path = Path(__file__).absolute().parent.parent.joinpath("README.md")
    readme = readme_path.read_text()
    for dataset in ["mipnerf360", "blender", "tanksandtemples"]:
        readme = update_dataset_results(readme, dataset)
    readme_path.write_text(readme)


if __name__ == "__main__":
    main()
