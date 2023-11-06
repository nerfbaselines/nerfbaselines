import click
from .train import train_command


@click.group()
def main():
    pass


main.add_command(train_command)
