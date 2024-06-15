import glob
import sys
from typing import Optional, Dict
import json
import subprocess
import os
import click
import logging
import shutil
from tempfile import TemporaryDirectory

from nerfbaselines.types import NB_PREFIX
from nerfbaselines.utils import setup_logging, handle_cli_error
from nerfbaselines.results import compile_dataset_results


def _get_repository_source():
    repo_path = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(repo_path, ".git")):
        logging.info(f"Found git repository at {repo_path}")
        return repo_path
    else:
        online_repo_path = "https://github.com/jkulhanek/nerfbaselines.git"
        logging.warning(f"Could not find git repository at {repo_path}, will use online repository: {online_repo_path}.")
        return online_repo_path


def _install_sandboxed_node(environ=None) -> Dict[str, str]:
    client_dir = os.path.join(NB_PREFIX, "nodeenv")

    def get_node_bin_dir() -> str:
        env_dir = os.path.join(client_dir, ".nodeenv")
        node_bin_dir = os.path.join(env_dir, "bin")
        if not os.path.exists(node_bin_dir):
            node_bin_dir = os.path.join(env_dir, "Scripts")
        return node_bin_dir

    node_bin_dir = get_node_bin_dir()
    if os.path.exists(os.path.join(node_bin_dir, "npx")):
        logging.info("Sandboxed nodejs is set up")

    else:
        env_dir = os.path.join(client_dir, ".nodeenv")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "nodeenv", "--node=20.4.0", env_dir]
            )
        except subprocess.CalledProcessError as e:
            logging.error("Failed to install sandboxed nodejs. Make sure nodeenv is installed.")
            raise click.exceptions.Exit(1) from e

        node_bin_dir = get_node_bin_dir()

    npx_path = os.path.join(node_bin_dir, "npx")
    assert os.path.join(npx_path), f"npx not found at {node_bin_dir}"

    subprocess_env = (environ if environ is not None else os.environ).copy()
    subprocess_env["NODE_VIRTUAL_ENV"] = str(os.path.dirname(node_bin_dir))
    subprocess_env["PATH"] = (
        node_bin_dir
        + (";" if sys.platform == "win32" else ":")
        + subprocess_env["PATH"]
    )
    return subprocess_env


@click.command("generate-web")
@click.option("--source", type=str, required=False, help="Path to the source directory containing all the results and checkpoint.")
@click.option("--output", "dest", type=str, required=True, help="Path to the output directory where the web page will be generated.")
@click.option("--sandbox-node", is_flag=True, help="Use a sandboxed node installation.")
@click.option("--datasets", type=str, required=False, help="A comma separated list of datasets to process.")
@click.option("--verbose", "-v", is_flag=True)
@handle_cli_error
def main(*,
         source: Optional[str] = None, 
         dest: str,
         sandbox_node: bool = False,
         datasets: Optional[str] = None,
         verbose: bool = False):
    setup_logging(verbose)
    if source is not None and not os.path.exists(source):
        raise RuntimeError(f"Source directory does not exist: {source}")
    if not os.path.exists(os.path.dirname(os.path.abspath(dest))):
        raise RuntimeError(f"Parent directory of the output directory does not exist: {os.path.dirname(dest)}")
    if os.path.exists(dest):
        raise RuntimeError(f"Output directory already exists: {dest}")
    logging.info(f"Generating web page from source directory: {source}")

    # Test if git works before starting
    try:
        subprocess.check_call(["git", "--version"])
    except subprocess.CalledProcessError:
        raise RuntimeError("git is not installed or not in the PATH")

    env = os.environ
    if sandbox_node:
        logging.info("Setting up sandboxed nodejs")
        env = _install_sandboxed_node(env)
    else:
        # Test if npm works before starting
        try:
            subprocess.check_call(["npm", "--version"])
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logging.error("npm is not installed or not in the PATH. Try using --sandbox-node")
            raise click.exceptions.Exit(1) from e
        logging.info("Using system nodejs")

    with TemporaryDirectory() as temp_dir:
        source_repo = _get_repository_source()
        logging.info(f"Cloning web template repository to {temp_dir}/web")
        # Clone repo
        subprocess.check_call(["git", "clone", source_repo, os.path.join(temp_dir, "web")])
        # Checkout the web branch
        subprocess.check_call(["git", "checkout", "web"], cwd=os.path.join(temp_dir, "web"))
        # Remove existing data directory
        logging.info("Removing existing data directory")
        shutil.rmtree(os.path.join(temp_dir, "web", "web", "data"), ignore_errors=True)
        os.makedirs(os.path.join(temp_dir, "web", "web", "data"), exist_ok=True)
        # Prepare data source
        if source is None:
            subprocess.check_call("git clone https://huggingface.co/jkulhanek/nerfbaselines".split() + [os.path.join(temp_dir, "results")], env={"GIT_LFS_SKIP_SMUDGE": "1"})
            source = os.path.join(temp_dir, "results")
        # Generate the data for the datasets
        if datasets is None:
            datasets = ",".join(set(os.path.split(x)[-1] for x in glob.glob(os.path.join(source, "*/*"))))
        logging.info(f"Generating data for datasets: {datasets}")
        for dataset in datasets.split(","):
            logging.info(f"Generating data for dataset: {dataset}")
            output = os.path.join(temp_dir, "web", "web", "data", f"{dataset}.json")
            dataset_info = compile_dataset_results(source, dataset)
            output_str = json.dumps(dataset_info, indent=2) + os.linesep
            with open(output, "w", encoding="utf8") as f:
                print(output_str, end="", file=f)
            logging.info(f"Generated web/data/{dataset}.json")
        # Generate the website
        logging.info("Generating the website")
        subprocess.check_call(["npm", "--yes", "install"], cwd=os.path.join(temp_dir, "web", "web"))
        subprocess.check_call(["npm", "--yes", "run", "build"], cwd=os.path.join(temp_dir, "web", "web"))
        # Copy the website to the output directory
        logging.info(f"Copying the website to the output directory: {os.path.abspath(dest)}")
        shutil.copytree(os.path.join(temp_dir, "web", "web", "out"), dest)

if __name__ == "__main__":
    main()

