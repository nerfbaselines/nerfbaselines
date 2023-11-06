import click
from .train import train_command
from .render import render_command


@click.group()
def main():
    pass


main.add_command(train_command)
main.add_command(render_command)
