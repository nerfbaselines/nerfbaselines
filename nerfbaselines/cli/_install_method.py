import types
import pprint
import logging
import os
import click
from nerfbaselines import (
    get_supported_methods,
    get_method_spec,
)
from nerfbaselines import backends
from nerfbaselines.io import open_any
from ._common import click_backend_option, NerfBaselinesCliCommand


def get_register_calls(file):
    # First, we collect all register() calls
    from nerfbaselines import _registry as registry
    register_calls = []
    with registry.collect_register_calls(register_calls):
        # Import the module from the file
        module = types.ModuleType("spec")
        if hasattr(file, "name"):
            module.__file__ = file.name
        file_content = file.read()
        if isinstance(file_content, bytes):
            file_content = file_content.decode('utf8')
        exec(file_content, module.__dict__)

    # Now we have aggregated all register() calls, we can process them
    output = []
    for spec in register_calls:
        spec_file_content = f"""from nerfbaselines import register
register({pprint.pformat(spec)})
"""
        output.append({
            "type": registry.get_spec_type(spec),
            "id": spec["id"],
            "spec": spec,
            "spec_file_content": spec_file_content
        })
    return output


@click.command("install", cls=NerfBaselinesCliCommand, short_help="Install a method or a spec file", help=(
    "Install either a pre-defined method or a custom method from a spec file. "
    "If installing from spec (`--spec` argument) - a python file with a register() call - the registered objects will be added to local registry. "
    "If installing a method (`--method` argument) the backend implementation for the method will be pre-installed. "
    "This also applies when specifying both `--spec` and `--backend` arguments."
))
@click.option("--method", type=click.Choice(list(get_supported_methods())), required=False, default=None, help="Method to install.")
@click.option("--spec", type=str, required=False, default=None, help="Path to a spec file (a Python file with `register()` calls) to install.")
@click.option("--force", is_flag=True, help="Overwrite existing specs. Only applies when `--spec` is provided.")
@click_backend_option()
def install_method_command(method, spec, backend_name, force=False):
    if method is not None:
        method_spec = get_method_spec(method)
        with backends.get_backend(method_spec, backend_name) as backend_impl:
            logging.info(f"Using method: {method}, backend: {backend_impl.name}")
            backend_impl.install()
    elif spec is not None:
        with open_any(spec, "r") as f:
            register_calls = get_register_calls(f)

        # Test if some of the specs are already registered
        from nerfbaselines import _registry as registry
        for register_call in register_calls:
            if register_call["type"] == "method" and register_call["id"] in registry.methods_registry:
                if not force:
                    raise RuntimeError(f"Method {register_call['id']} is already registered")
                else:
                    logging.warning(f"Method {register_call['id']} is already registered, but --force was provided")
            if register_call["type"] == "dataset" and register_call["id"] in registry.datasets_registry:
                if not force:
                    raise RuntimeError(f"Dataset {register_call['id']} is already registered")
                else:
                    logging.warning(f"Dataset {register_call['id']} is already registered, but --force was provided")
            if register_call["type"] == "dataset_loader" and register_call["id"] in registry.dataset_loaders_registry:
                if not force:
                    raise RuntimeError(f"Dataset loader {register_call['id']} is already registered")
                else:
                    logging.warning(f"Dataset loader {register_call['id']} is already registered, but --force was provided")
            if register_call["type"] == "evaluation_protocol" and register_call["id"] in registry.evaluation_protocols_registry:
                if not force:
                    raise RuntimeError(f"Evaluation protocol {register_call['id']} is already registered")
                else:
                    logging.warning(f"Evaluation protocol {register_call['id']} is already registered, but --force was provided")

        # Register the specs
        for register_call in register_calls:
            output_name = f"{register_call['type']}-{register_call['id']}.py"
            os.makedirs(registry.METHOD_SPECS_PATH, exist_ok=True)
            with open(os.path.join(registry.METHOD_SPECS_PATH, output_name), "w", encoding='utf8') as f:
                f.write(register_call["spec_file_content"])
            logging.info(f"Spec file {output_name} saved to {registry.METHOD_SPECS_PATH}")

        # If the backend_name was supplied from the command line, install the backend
        # Test click.get_current_context() param source
        for call in register_calls:
            if call["type"] != "method":
                continue
            if click.get_current_context().get_parameter_source("backend_name") == click.core.ParameterSource.COMMANDLINE:
                with backends.get_backend(call["spec"], backend_name) as backend_impl:
                    logging.info(f"Using backend: {backend_impl.name} for method: {call['id']}")
                    backend_impl.install()
    else:
        raise RuntimeError("Either --method or --spec must be provided")

