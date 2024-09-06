import sys
from click import Command
import pytest
from unittest import mock


@pytest.mark.parametrize("args", [
    "fix-checkpoint --checkpoint checkpoint --data data --new-checkpoint new_checkpoint".split(),
    "train --data data --method zipnerf --checkpoint checkpoint --output output".split(),
    "render --data data --checkpoint checkpoint --output output".split(),
    "render-trajectory --checkpoint checkpoint --output output --trajectory trajectory".split(),
    "evaluate predictions --output output".split(),
    "build-docker-image".split(),
    "download-dataset dataset".split(),
    "generate-dataset-results".split(),
    "web build --output output".split(),
    "web dev".split(),
    "viewer".split(),
])
def test_cli_command(args):
    if args[0] == "viewer" and sys.version_info < (3, 8):
        pytest.skip("Viewer command is not supported on Python < 3.8")
    from nerfbaselines.cli import main

    new_invoke = mock.Mock()

    with mock.patch("sys.argv", ["nerfbaselines"] + args), mock.patch.object(Command, "invoke", new_invoke), mock.patch("sys.exit") as exit:
        main()
    assert new_invoke.called
    assert exit.called
    assert exit.call_args[0][0] == 0

    # ctx: Context = new_invoke.call_args[0][0]
    # print(ctx)
    # print(dir(ctx))
    # print(ctx.params)
    # assert ctx ==0

