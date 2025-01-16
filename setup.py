#!/usr/bin/env python
# This file downloads the required resources for the package.
# It is run both for sdist and bdist, and the resources are included in the package.
import os
import urllib.request
import setuptools


_LPIPS_WEIGHTS_URL = "https://github.com/richzhang/PerceptualSimilarity/raw/c33f89e9f46522a584cf41d8880eb0afa982708b/lpips/weights/v{version}/{net}.pth"
_pull_data = [
    (_LPIPS_WEIGHTS_URL.format(version=version, net=net), 'nerfbaselines/_lpips_weights/{net}-{version}.pth'.format(version=version, net=net))
    for version, net in [
        ('0.1', 'vgg'),
        ('0.1', 'alex'),
    ]
]


def _ensure_files_exist():
    root = os.path.dirname(__file__)
    for url, path in _pull_data:
        local_path = os.path.join(root, path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        if not os.path.exists(local_path):
            urllib.request.urlretrieve(url, local_path)


if __name__ == "__main__":
    _ensure_files_exist()
    setuptools.setup()
