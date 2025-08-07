import re
import os
import subprocess
import shlex
from typing import Optional
import hashlib

from nerfbaselines import get_supported_methods, get_method_spec, NB_PREFIX
from ._rpc import RemoteProcessRPCBackend, get_safe_environment
try:
    from typing import TypedDict, Required
except ImportError:
    from typing_extensions import TypedDict, Required


def shlex_join(split_command):
    return ' '.join(shlex.quote(arg) for arg in split_command)


class CondaBackendSpec(TypedDict, total=False):
    environment_name: Required[str]
    python_version: Optional[str]
    install_script: Optional[str]


def conda_get_environment_hash(spec: CondaBackendSpec):
    value = hashlib.sha256()

    def maybe_update(x):
        if x is not None:
            value.update(x.encode("utf8"))

    maybe_update(spec.get("python_version"))
    maybe_update(spec.get("install_script"))
    return value.hexdigest()


def conda_get_install_script(spec: CondaBackendSpec, package_path: Optional[str] = None, environment_path: Optional[str] = None):
    environment_name = spec.get("environment_name")
    assert environment_name is not None, "CondaBackend requires environment_name to be specified"
    environment_hash = conda_get_environment_hash(spec)

    custom_environment_path = True
    if environment_path is None:
        custom_environment_path = False
        environments_path = os.environ.get("NERFBASELINES_CONDA_ENVIRONMENTS", os.path.join(NB_PREFIX, "conda-envs"))
        environment_path = os.path.join(environments_path, environment_name, environment_hash, environment_name)
    env_path = environment_path
    args = []
    python_version = spec.get("python_version")
    if python_version is not None:
        args.append(f"python={python_version}")
    install_dependencies_script = ''

    def is_method_allowed(method):
        spec = get_method_spec(method)
        conda_spec = spec.get("conda")
        if conda_spec is not None:
            return conda_get_environment_hash(conda_spec) == environment_hash
        return False

    allowed_methods = ",".join((k for k in get_supported_methods() if is_method_allowed(k)))
    script = "set -eo pipefail\n"

    if not custom_environment_path:
        script += f"""
# Clear old environments
was_message=0
if [ -d {shlex.quote(os.path.dirname(os.path.dirname(env_path)))} ]; then
    for hash in $(ls -1 {shlex.quote(os.path.dirname(os.path.dirname(env_path)))}); do
        if [ "$hash" != {shlex.quote(environment_hash)} ]; then
            if [ "$was_message" -eq 0 ]; then
                echo "Clearing old environments"
                was_message=1
            fi
            echo "   Removing old environment $hash"
            rm -rf {shlex.quote(os.path.dirname(os.path.dirname(env_path)))}"/$hash"
        fi
    done
fi
"""
    prepare_default_nerfbaselines = ""
    if package_path is not None:
        prepare_default_nerfbaselines = f"""
# Prepare default nerfbaselines
site_packages="$(python -c 'import site; print(site.getsitepackages()[0])')"
if [[ "$site_packages" != "$CONDA_PREFIX/"* ]]; then
    echo "ERROR: site-packages is not in the conda environment"; exit 1;
fi
ln -s {shlex.quote(package_path)} "$site_packages"
"""
    prepare_default_nerfbaselines += f"""
# Add nerfbaselines to the path
if ! nerfbaselines >/dev/null 2>&1; then
    echo '#!/usr/bin/env python3\nfrom nerfbaselines.__main__ import main; main()'>"$CONDA_PREFIX/bin/nerfbaselines"
    chmod +x "$CONDA_PREFIX/bin/nerfbaselines"
fi
"""
    script += f"""
# Create new environment
eval "$(conda shell.bash hook)"
mkdir -p {shlex.quote(os.path.dirname(env_path))}
if [ ! -e {shlex.quote(env_path + ".ack.txt")} ]; then
if [ -e {shlex.quote(env_path + ".lock/pid")} ] && [ ! -e /proc/$(cat {shlex.quote(env_path + ".lock/pid")}) ] && [ "$(cat {shlex.quote(env_path + ".lock/nodename")})" = "$(hostname)" ]; then
    echo "Removing stale lock file {shlex.quote(env_path + ".lock")} (no process running)"
    rm -rf {shlex.quote(env_path + ".lock")}
fi
if ! mkdir {shlex.quote(env_path + ".lock")} 2>/dev/null; then
    # Wait for the other process to write PID
    sleep 1
    pid=$(cat {shlex.quote(env_path + ".lock/pid")})
    nodename=$(cat {shlex.quote(env_path + ".lock/nodename")})
    echo "Creating new is already running (lock: {shlex.quote(env_path + ".lock")},pid:$pid,nodename:$nodename)" >&2
    echo "Waiting for the other process to finish" >&2
    while [ -e {shlex.quote(env_path + ".lock")} ] || [ -e /proc/$pid ]; do
        sleep 1
    done
    if [ ! -e {shlex.quote(env_path + ".ack.txt")} ]; then
        echo "The other process failed to build the environment." >&2
        exit 1
    else
        echo "The other process finished building the environment." >&2
        exit 0
    fi
fi
echo $(hostname) > {shlex.quote(env_path + ".lock/nodename")}
echo $$ > {shlex.quote(env_path + ".lock/pid")}
trap "rm -rf {shlex.quote(env_path + ".lock")}" EXIT
echo "Creating new environment {shlex.quote(env_path)}"

rm -rf {shlex.quote(env_path)}
conda deactivate
{shlex_join(["conda", "create", "--prefix", env_path, "-y", "-c", "conda-forge", "--override-channels"] + args)}
conda activate {shlex.quote(env_path)}
echo -e 'channels:\n  - conda-forge\n' > {shlex.quote(os.path.join(env_path, ".condarc"))}
conda install -y pip conda-build
pip install --upgrade pip setuptools
mkdir -p {shlex.quote(os.path.join(env_path, "nb-sources"))}
mkdir -p {shlex.quote(os.path.join(env_path, "src"))}
cd {shlex.quote(os.path.join(env_path, "src"))}
{spec.get('install_script') or ''}
{install_dependencies_script}
{prepare_default_nerfbaselines}
echo '#!/bin/bash' > {shlex.quote(os.path.join(env_path, ".activate.sh"))}
echo 'eval "$(conda shell.bash hook)"' >> {shlex.quote(os.path.join(env_path, ".activate.sh"))}
echo 'conda activate {shlex.quote(env_path)};export NERFBASELINES_BACKEND=python;export NERFBASELINES_ALLOWED_METHODS="{allowed_methods}"' >> {shlex.quote(os.path.join(env_path, ".activate.sh"))}
echo 'exec "$@"' >> {shlex.quote(os.path.join(env_path, ".activate.sh"))}
chmod +x {shlex.quote(os.path.join(env_path, ".activate.sh"))}
touch {shlex.quote(env_path + ".ack.txt")}
echo "0" > {shlex.quote(env_path + ".ack.txt")}
fi
"""
    return script

class CondaBackend(RemoteProcessRPCBackend):
    name = "conda"

    def __init__(self, spec: CondaBackendSpec):
        self._spec = spec
        super().__init__(python_path="python")
        assert self._spec.get("environment_name") is not None, "CondaBackend requires environment_name to be specified"

    def _prepare_package_path(self, env_path):
        os.makedirs(os.path.join(env_path, "nb-sources"), exist_ok=True)
        # Sanitize env path
        package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # NOTE: There was path-too-long error
        # safe_package_path = re.sub("[^0-9a-zA-Z_]", "_", os.path.abspath(package_path))
        safe_package_path = hashlib.sha256(package_path.encode("utf8")).hexdigest()[:16]
        target = os.path.join(env_path, "nb-sources", safe_package_path, "nerfbaselines")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if not os.path.exists(target):
            os.symlink(package_path, target)
        return os.path.dirname(target)


    def _launch_worker(self, args, env):
        environments_path = os.environ.get("NERFBASELINES_CONDA_ENVIRONMENTS", os.path.join(NB_PREFIX, "conda-envs"))
        environment_name = self._spec.get("environment_name")
        assert environment_name is not None, "CondaBackend requires environment_name to be specified"
        env_path = os.path.join(environments_path, environment_name, conda_get_environment_hash(self._spec), environment_name)
        env["PYTHONPATH"] = self._prepare_package_path(env_path)
        args = [os.path.join(env_path, ".activate.sh")] + list(args)
        return super()._launch_worker(args, env)

    def install(self):
        package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        subprocess.check_call(["bash", "-c", conda_get_install_script(self._spec, package_path=package_path)])

    def shell(self, args=None):
        environments_path = os.environ.get("NERFBASELINES_CONDA_ENVIRONMENTS", os.path.join(NB_PREFIX, "conda-envs"))
        environment_name = self._spec.get("environment_name")
        assert environment_name is not None, "CondaBackend requires environment_name to be specified"
        env_path = os.path.join(environments_path, environment_name, conda_get_environment_hash(self._spec), environment_name)
        env = get_safe_environment()
        env["PYTHONPATH"] = self._prepare_package_path(env_path)
        args = ["bash"] if args is None else list(args)
        args = [os.path.join(env_path, ".activate.sh")] + args
        os.execvpe(args[0], args, env)
