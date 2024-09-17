import os
WEBPAGE_URL = "https://nerfbaselines.github.io"
DOCKER_REPOSITORY = "ghcr.io/nerfbaselines/nerfbaselines"
NB_PREFIX = os.path.expanduser(os.environ.get("NERFBASELINES_PREFIX", "~/.cache/nerfbaselines"))

CODE_REPOSITORY = "github.com/nerfbaselines/nerfbaselines"
RESULTS_REPOSITORY = "huggingface.co/jkulhanek/nerfbaselines"
DATASETS_REPOSITORY = "huggingface.co/datasets/jkulhanek/nerfbaselines-data"
SUPPLEMENTARY_RESULTS_REPOSITORY = "huggingface.co/datasets/jkulhanek/nerfbaselines-supplementary"
del os
