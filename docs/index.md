<div style="display:flex;flex-direction:row;justify-content:center;align-items:center">
<img src="_static/logo.png" style="margin-right: 1.0em;width:90px;height:90px" />
<div style="display:flex;flex-direction:column">

# NerfBaselines documentation

<div style="margin-top:-1em">

[![PyPI - Version](https://img.shields.io/pypi/v/nerfbaselines)](https://pypi.org/project/nerfbaselines/)
[![GitHub License](https://img.shields.io/badge/license-MIT-%2397ca00)](https://github.com/nerfbaselines/nerfbaselines/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/nerfbaselines)](https://pepy.tech/project/nerfbaselines)

</div>
</div>
</div>

NerfBaselines is a framework for **evaluating and comparing existing NeRF and 3DGS methods**. Currently, most official implementations use different dataset loaders, evaluation protocols, and metrics, which renders benchmarking difficult. Therefore, this project aims to provide a **unified interface** for running and evaluating methods on different datasets in a consistent way using the same metrics. But instead of reimplementing the methods, **we use the official implementations** and wrap them so that they can be run easily using the same interface.

Please visit the <a href="https://nerfbaselines.github.io/">project page to see the results</a> of implemented methods on dataset benchmarks.<br/>

<h3 style="margin-top:1em;text-align:center">
<a href="https://nerfbaselines.github.io/"><img style="height:1em;position:relative;top:0.12em" src='data:image/svg+xml;charset=utf-8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="2 2 21 21" fill="none"  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M3 12a9 9 0 1 0 18 0a9 9 0 0 0 -18 0" /><path d="M3.6 9h16.8" /><path d="M3.6 15h16.8" /><path d="M11.5 3a17 17 0 0 0 0 18" /><path d="M12.5 3a17 17 0 0 1 0 18" /></svg>' /> Web</a> &nbsp;&nbsp;|&nbsp;&nbsp;
<a href="https://github.com/nerfbaselines/nerfbaselines"><img style="height:1em;position:relative;top:0.12em" src='data:image/svg+xml;charset=utf-8,<svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path></svg>' /> GitHub</a> &nbsp;&nbsp;|&nbsp;&nbsp;
<a href="https://arxiv.org/pdf/2406.17345.pdf"><img style="height:1em;position:relative;top:0.12em" src='data:image/svg+xml;charset=utf-8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M12 2l.117 .007a1 1 0 0 1 .876 .876l.007 .117v4l.005 .15a2 2 0 0 0 1.838 1.844l.157 .006h4l.117 .007a1 1 0 0 1 .876 .876l.007 .117v9a3 3 0 0 1 -2.824 2.995l-.176 .005h-10a3 3 0 0 1 -2.995 -2.824l-.005 -.176v-14a3 3 0 0 1 2.824 -2.995l.176 -.005h5z" /><path d="M19 7h-4l-.001 -4.001z" /></svg>' /> Paper</a>
</h3>

## Main features
- **Unified interface**: All methods can be run using the same interface.
- **Consistent evaluation**: All methods are evaluated using the same metrics and protocols.
- **Reproducibility**: All methods are run using the official implementations.
- **Easy to use**: The CLI is easy to use and requires minimal setup.
- **Extensible**: New methods can be added easily by wrapping the official implementation.
- **Public benchmarks**: The results of all methods (and checkpoints) are available on [the website](https://nerfbaselines.github.io/).

For the full list of implemented methods, see the [methods](methods.md) section.
For the full list of available datasets (datasets which support automatic download), see the [datasets](datasets.md) section.

```{tip}
NerfBaselines now supports online demos for 3DGS-based methods. Check out the demos on the [benchmark page](https://nerfbaselines.github.io/)!
```

## Contents
The documentation is organized into several sections with increasing level of detail and difficulty:
```{toctree}
:maxdepth: 1
installation
quickstart
using-custom-data
viewer
python-tutorial
```

```{toctree}
:maxdepth: 1
:caption: Advanced
adding-new-methods
custom-dataset-loader
adding-new-datasets
backends
exporting-demos
exporting-meshes
exporting-results-tables
custom-web
```

```{toctree}
:maxdepth: 1
:caption: Reference

Methods <methods>
Datasets <datasets>
CLI <cli>
API <api/modules>
```

```{tip}
The documentation is available for all released versions of the project. You can switch between versions using the version selector in the bottom left corner.
```

## Acknowledgements
A big thanks to the authors of all implemented methods for the great work they have done.
We would also like to thank the authors of [NerfStudio](https://github.com/nerfstudio-project/nerfstudio).
This work was supported by the Czech Science Foundation (GAČR) EXPRO (grant no. 23-07973X), the Grant Agency of the Czech Technical University in Prague (grant no. SGS24/095/OHK3/2T/13), 
and by the Ministry of Education, Youth and Sports of the Czech Republic through the e-INFRA CZ (ID:90254).

## Citation
If you use this project in your research, please cite the [following paper](https://arxiv.org/pdf/2406.17345.pdf):
```bibtex
@article{kulhanek2024nerfbaselines,
  title={{N}erf{B}aselines: Consistent and Reproducible Evaluation of Novel View Synthesis Methods},
  author={Jonas Kulhanek and Torsten Sattler},
  year={2024},
  journal={arXiv},
}
```
