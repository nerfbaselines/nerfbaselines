import logging
import click
from ._common import NerfBaselinesCliCommand
import nerfbaselines


@click.command("build-docker-image", hidden=True, cls=NerfBaselinesCliCommand)
@click.option("--method", type=click.Choice(list(nerfbaselines.get_supported_methods("docker"))), required=False)
@click.option("--environment", type=str, required=False)
@click.option("--skip-if-exists-remotely", is_flag=True)
@click.option("--tag-latest", is_flag=True)
@click.option("--push", is_flag=True)
def build_docker_image_command(method=None, environment=None, push=False, skip_if_exists_remotely=False, tag_latest=False):
    from nerfbaselines.backends._docker import build_docker_image, get_docker_spec

    spec = None
    if method is not None:
        spec = nerfbaselines.get_method_spec(method)
        if spec is None:
            raise RuntimeError(f"Method {method} not found")
        spec = get_docker_spec(spec)
        if spec is None:
            raise RuntimeError(f"Method {method} does not support building docker images")
        env_name = spec["environment_name"]
        logging.info(f"Building docker image for environment {env_name} (from method {method})")
    elif environment is not None:
        for method in nerfbaselines.get_supported_methods("docker"):
            spec = nerfbaselines.get_method_spec(method)
            spec = get_docker_spec(spec)
            if spec is None:
                continue
            if spec.get("environment_name") == environment:
                break
        if spec is None:
            raise RuntimeError(f"Environment {environment} not found")
        logging.info(f"Building docker image for environment {environment}")
    else:
        logging.info("Building base docker image")
    build_docker_image(spec, skip_if_exists_remotely=skip_if_exists_remotely, push=push, tag_latest=tag_latest)


