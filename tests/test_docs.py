import os
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


# TODO: Make tests work on windows in GH actions (currently firewall issue in CI)
@pytest.mark.skipif(sys.platform == "win32" and os.getenv("CI") == "true", reason="CI firewall windows issue")
def test_export_results_table_tutorial_markdown():
    _call_cli("generate-dataset-results --output-type markdown --dataset mipnerf360".split())
    _call_cli("generate-dataset-results --output-type markdown --dataset mipnerf360 --scenes garden,bonsai".split())
    _call_cli("generate-dataset-results --output-type markdown --dataset mipnerf360 --method-links website".split())
        
@pytest.mark.skipif(sys.platform == "win32" and os.getenv("CI") == "true", reason="CI firewall windows issue")
def test_export_results_table_tutorial_latex():
    _call_cli("generate-dataset-results --output-type latex --dataset mipnerf360".split())
    _call_cli("generate-dataset-results --output-type latex --dataset mipnerf360 --scenes garden,bonsai".split())

