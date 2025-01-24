import copy
import itertools
import importlib
import click
from ._web import web_click_group


class LazyGroup(click.Group):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._lazy_commands = dict()

    def get_command(self, ctx, cmd_name):
        cmd_def = self._lazy_commands.get(cmd_name, None)
        package = cmd_def.get("command", None) if cmd_def is not None else None
        if package is not None:
            if isinstance(package, str):
                fname = "main"
                if ":" in package:
                    package, fname = package.split(":")
                package = getattr(importlib.import_module(package, __name__), fname)
            command = copy.deepcopy(package)
            command.name = cmd_name
            command.hidden = cmd_def.get("hidden", False)
            return command
        return super().get_command(ctx, cmd_name)

    def list_commands(self, ctx):
        del ctx
        return sorted(itertools.chain(self._lazy_commands.keys(), self.commands.keys()))

    def add_lazy_command(self, package_name: str, command_name: str, hidden=False):
        self._lazy_commands[command_name] = dict(
            command=package_name,
            hidden=hidden,
        )

    def format_commands(self, ctx, formatter) -> None:
        """Extra format methods for multi methods that adds all the commands
        after the options.
        """
        # allow for 3 times the default spacing
        commands = []
        lazy_cmds = ((k, v) for k, v in self._lazy_commands.items() if not v["hidden"])
        for name, cmd in sorted(itertools.chain(lazy_cmds, self.commands.items()), key=lambda x: x[0]):
            if isinstance(cmd, click.Group):
                for cmd2 in cmd.list_commands(ctx):
                    sub_cmd = cmd.get_command(ctx, cmd2)
                    if sub_cmd is not None:
                        commands.append(" ".join((name, cmd2)))
            else:
                commands.append(name)

        if len(commands):
            # limit = formatter.width - 6 - max(len(cmd[0]) for cmd in commands)
            rows = []
            for subcommand in commands:
                rows.append((subcommand, ""))

            with formatter.section("Commands"):
                formatter.write_dl(rows)


@click.group(name="nerfbaselines", cls=LazyGroup)
def main():
    pass


main.add_command(web_click_group)
main.add_lazy_command("nerfbaselines.cli._shell:shell_command", "shell")
main.add_lazy_command("nerfbaselines.cli._export_demo", "export-demo")
main.add_lazy_command("nerfbaselines.cli._export_mesh:export_mesh_command", "export-mesh")
main.add_lazy_command("nerfbaselines.cli._test_method", "test-method")
main.add_lazy_command("nerfbaselines.cli._render:render_command", "render")
main.add_lazy_command("nerfbaselines.cli._render:render_trajectory_command", "render-trajectory")
main.add_lazy_command("nerfbaselines.cli._generate_dataset_results:main", "generate-dataset-results")
main.add_lazy_command("nerfbaselines.cli._fix_checkpoint:main", "fix-checkpoint")
# nerfbaselines install-method is deprecated, but we keep it for compatibility
main.add_lazy_command("nerfbaselines.cli._install_method:install_method_command", "install-method", hidden=True)
main.add_lazy_command("nerfbaselines.cli._install_method:install_method_command", "install")
main.add_lazy_command("nerfbaselines.cli._fix_output_artifact:main", "fix-output-artifact")
main.add_lazy_command("nerfbaselines.cli._train:train_command", "train")
main.add_lazy_command("nerfbaselines.cli._viewer:viewer_command", "viewer")
main.add_lazy_command("nerfbaselines.cli._build_docker_image:build_docker_image_command", "build-docker-image", hidden=True)
main.add_lazy_command("nerfbaselines.cli._download_dataset:download_dataset_command", "download-dataset")
main.add_lazy_command("nerfbaselines.cli._evaluate:evaluate_command", "evaluate")
main.add_lazy_command("nerfbaselines.cli._measure_fps:measure_fps_command", "measure-fps")
