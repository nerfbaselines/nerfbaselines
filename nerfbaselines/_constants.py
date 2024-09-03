import os
REPOSITORY_NAME = "jkulhanek/nerfbaselines"
WEBPAGE_URL = "https://jkulhanek.com/nerfbaselines"
DOCKER_REPOSITORY = "ghcr.io/jkulhanek/nerfbaselines"
NB_PREFIX = os.path.expanduser(os.environ.get("NERFBASELINES_PREFIX", "~/.cache/nerfbaselines"))
del os
