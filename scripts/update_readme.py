import os
import sys
from pathlib import Path
import click

try:
    import nerfbaselines  # noqa
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    import nerfbaselines  # noqa
from nerfbaselines.cli._generate_dataset_results import main as generate_dataset_results_command
from nerfbaselines.results import get_dataset_info
from nerfbaselines._constants import WEBPAGE_URL
import tempfile


def update_licenses(readme: str):
    from nerfbaselines import get_supported_methods, get_method_spec
    lines = readme.rstrip(os.linesep).splitlines()

    def simplify(s: str):
        return s.lower().replace(" ", "").replace("-", "").replace("_", "")

    # Locate old section
    section_start = next((x for x in range(len(lines)) if simplify(f"For the currently implemented methods, the following licenses apply:") in simplify(lines[x])), None)
    if section_start is None:
        raise RuntimeError(f"Could not locate licenses section in README.md")
    section_end = next((x for x in range(section_start + 1, len(lines)) if lines[x].startswith("##")), len(lines))

    # Replace old results with new results
    methods_licenses = []
    for method in get_supported_methods():
        spec = get_method_spec(method).get("metadata", {})
        if ":" in method:
            continue
        if spec.get("licenses"):
            licenses = ", ".join(["[{name}]({url})".format(**x) if "url" in x else x["name"] for x in spec["licenses"]])
            method_name = spec.get("name", method)
            methods_licenses.append(f"- {method_name}: {licenses}")
    methods_licenses.sort(key=lambda x: x.lower())
    new_section = f"""{lines[section_start]}
{os.linesep.join(methods_licenses)}

"""
    return os.linesep.join(lines[:section_start] + [new_section] + lines[section_end:]) + os.linesep


def update_reproducing_results_table(readme: str):
    from nerfbaselines import get_supported_methods, get_method_spec
    from nerfbaselines import get_supported_datasets, get_dataset_spec
    lines = readme.rstrip(os.linesep).splitlines()

    def simplify(s: str):
        return s.lower().replace(" ", "").replace("-", "").replace("_", "")

    # Locate old section
    section_start = next((x for x in range(len(lines)) if simplify("## Implementation status") in simplify(lines[x])), None)
    if section_start is None:
        raise RuntimeError(f"Could not locate Reproducing results section in README.md")
    section_end = next((x for x in range(section_start + 1, len(lines)) if lines[x].startswith("##")), len(lines))

    # Replace old results with new results
    labels = {
        "working-not-reproducing": "🥈 silver",
        "working": "🥇 gold",
        "reproducing": "🥇 gold",
        "not-working": "❌",
        None: "❔",
    }
    max_label_length = 1 + max(len(x) for x in labels.values())
    max_method_length = max(len("Method"), max(len(get_method_spec(method).get("metadata", {}).get("name", "")) for method in get_supported_methods()))
    out = ""
    datasets = sorted([x for x in get_supported_datasets()], key=lambda x: get_dataset_spec(x).get("metadata", {}).get("name").lower())
    dataset_names = [get_dataset_spec(x).get("metadata", {}).get("name") for x in datasets]
    out += f"| {'Method'.ljust(max_method_length)} "
    for i, dataset in enumerate(datasets):
        max_column_length = max(max_label_length, len(dataset_names[i]))
        out += f"| {dataset_names[i].ljust(max_column_length)} "
    out += "|\n"
    out += f"|:{''.ljust(max_method_length, '-')} "
    for i, dataset in enumerate(datasets):
        max_column_length = max(max_label_length, len(dataset_names[i]))
        out += f"|:{''.ljust(max_column_length, '-')} "
    out += "|\n"
    for method in sorted(get_supported_methods(), key=lambda x: get_method_spec(x).get("metadata", {}).get("name", x).lower()):
        spec = get_method_spec(method)
        if ":" in method:
            continue
        method_name = spec.get("metadata", {}).get("name", None)
        if not method_name:
            continue
        out += f"| {method_name.ljust(max_method_length)} "
        impl_status = spec.get("implementation_status", {})
        for i, dataset in enumerate(datasets):
            max_column_length = max(max_label_length, len(dataset_names[i]))
            out += f"| {labels.get(impl_status.get(dataset, None)).ljust(max_column_length-1)} "
        out += "|\n"
    new_section = f"""{lines[section_start]}
{out}

"""
    return os.linesep.join(lines[:section_start] + [new_section] + lines[section_end:]) + os.linesep



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
    readme = update_reproducing_results_table(readme)
    for dataset in ["mipnerf360", "blender", "tanksandtemples"]:
        readme = update_dataset_results(readme, dataset)
    readme = update_licenses(readme)
    readme_path.write_text(readme)


if __name__ == "__main__":
    main()
