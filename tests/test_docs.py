import sys
import pytest


def _call_cli(args):
    from nerfbaselines.cli import main

    old_argv = sys.argv
    try:
        sys.argv = args
        main(args=args)
    except SystemExit as e:
        if e.code != 0:
            raise
    finally:
        sys.argv = old_argv


def test_export_results_table_tutorial_markdown():
    _call_cli("generate-dataset-results --output-type markdown --dataset mipnerf360".split())
    _call_cli("generate-dataset-results --output-type markdown --dataset mipnerf360 --scenes garden,bonsai".split())
    _call_cli("generate-dataset-results --output-type markdown --dataset mipnerf360 --method-links website".split())
        

def test_export_results_table_tutorial_latex():
    _call_cli("generate-dataset-results --output-type latex --dataset mipnerf360".split())
    _call_cli("generate-dataset-results --output-type latex --dataset mipnerf360 --scenes garden,bonsai".split())

