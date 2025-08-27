# rpcbf-safety-on-the-fly

[![Alt text for image](assets/paper.png)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11122656)

Code accompanying the paper: "Safety on the Fly: Constructing Robust Safety Filters via Policy Control Barrier Functions at Runtime", accepted in IEEE Robotics and Automation Letters. For further details, please refer to the [paper](https://ieeexplore.ieee.org/document/11122656) or the [website](https://oswinso.xyz/rpcbf/).



## Installation
This project uses poetry for dependency management. To install poetry, run the following:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```
More information on poetry can be found [here](https://python-poetry.org/docs/).

Run the following in the root directory of the project:
```bash
poetry install
```
This will install all the dependencies required for the project in a virtual environment (`poetry env info` to get information on where it was saved).
Source the virtual environment:
```bash
 source {cache-dir}/virtualenvs/.venv/bin/activate
```

## Quickstart
```bash
python3 scripts/try_dbint_rpcbf.py
```

## Citing this work
```bibtex
@article{knoedler2025safety,
  title={Safety on the Fly: Constructing Robust Safety Filters via Policy Control Barrier Functions at Runtime},
  author={Knoedler, Luzia and So, Oswin and Yin, Ji and Black, Mitchell and Serlin, Zachary and Tsiotras, Panagiotis and Alonso-Mora, Javier and Fan, Chuchu},
  journal={IEEE Robotics and Automation Letters},
  year={2025},
  publisher={IEEE}
}
```

## Troubleshooting
If you run into problems of any kind, don't hesitate to open an [issue](https://github.com/LuziaKn/rpcbf-safety-on-the-fly/issues) on this repository.