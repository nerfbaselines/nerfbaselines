#!/usr/bin/env python
# This file downloads the required resources for the package.
# It is run both for sdist and bdist, and the resources are included in the package.
import os
import urllib.request
import setuptools.command.sdist
try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:
    _bdist_wheel = None  # type: ignore


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


cmdclass = {
    'sdist': sdist,
}

if _bdist_wheel is not None:
    class bdist_wheel(_bdist_wheel):
        def run(self):
            # generate data files
            _ensure_files_exist()
            super().run()

    cmdclass['bdist_wheel'] = bdist_wheel  # type: ignore


if __name__ == "__main__":
    setuptools.setup(
        data_files=[
            ('generated', [x[1] for x in _pull_data]),
        ],
        cmdclass=cmdclass,
    )
