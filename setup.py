#!/usr/bin/env python
# This file downloads the required resources for the package.
# It is run both for sdist and bdist, and the resources are included in the package.
import os
import setuptools.command.sdist
import setuptools.command.bdist_wheel
import urllib.request


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


class sdist(setuptools.command.sdist.sdist):
    def run(self):
        # generate data files
        _ensure_files_exist()
        super().run()


class bdist_wheel(setuptools.command.bdist_wheel.bdist_wheel):
    def run(self):
        # generate data files
        _ensure_files_exist()
        super().run()


if __name__ == "__main__":
    setuptools.setup(
        data_files=[
            ('generated', [x[1] for x in _pull_data]),
        ],
        cmdclass={
            'sdist': sdist,
            'bdist_wheel': bdist_wheel
        },
    )
