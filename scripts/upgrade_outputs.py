from pathlib import Path
import argparse
import zipfile
import logging


def upgrade_nb_info(nb_info):
    pass


def upgrade_outputs(path: Path):
    assert path.exists()

    with zipfile.ZipFile(str(path), "w") as zf:
        for member in zf.infolist():
            print(member)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    upgrade_outputs(args.path)