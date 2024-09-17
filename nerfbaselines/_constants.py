import os
WEBPAGE_URL = "https://nerfbaselines.github.io"
DOCKER_REPOSITORY = "ghcr.io/jkulhanek/nerfbaselines"
NB_PREFIX = os.path.expanduser(os.environ.get("NERFBASELINES_PREFIX", "~/.cache/nerfbaselines"))

CODE_REPOSITORY = "github.com/jkulhanek/nerfbaselines"
RESULTS_REPOSITORY = "huggingface.co/jkulhanek/nerfbaselines"
DATASETS_REPOSITORY = "huggingface.co/datasets/jkulhanek/nerfbaselines-data"
SUPPLEMENTARY_RESULTS_REPOSITORY = "huggingface.co/datasets/jkulhanek/nerfbaselines-supplementary"
del os
