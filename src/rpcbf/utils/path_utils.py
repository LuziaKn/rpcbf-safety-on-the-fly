import pathlib


def get_root_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent.parent.parent


def get_plot_dir() -> pathlib.Path:
    plot_dir = get_root_dir() / "plots"
    plot_dir.mkdir(exist_ok=True)
    return plot_dir

def get_data_dir() -> pathlib.Path:
    plot_dir = get_root_dir() / "data"
    plot_dir.mkdir(exist_ok=True)
    return plot_dir
