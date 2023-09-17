"""
Export plots to image (PNG, JPEG) or PDF files.

---------------------------------------------------------------------

Requires Python packages/modules:
  -  :mod:`sympy`
  -  :mod:`matplotlib`

---------------------------------------------------------------------
"""
# Library
import warnings
import logging
from os.path import exists, join, realpath
from os import mkdir, PathLike
from json import dump
from typing import Dict, Tuple, Union, Optional, List

# SymPy
from sympy import latex

# MatPlotLib
from matplotlib.pyplot import figure

warnings.filterwarnings("ignore")

__all__ = [
    "create_dir",
    "create_directories",
    "export_results",
    "export_plots",
    "export_plot",
]


def create_directories(
    results_path: Tuple[str, str] = ("..", "Results"), 
    results_dir: str = "Demo"
) -> str:
    """
    Create results parent and target directory.

    Args:
        results_path: path to parent results directory
            (to be created if necessary)
        results_dir: target results directory (to be created)

    Returns:
        str: path to target results directory (see :mod:`os.path`)
    """
    results_path_ = ["."] + list(results_path)
    create_dir(join(*results_path_))
    results_dir_ = results_path_ + [results_dir]
    return create_dir(join(*results_dir_))


def create_dir(dir_: str) -> str:
    """
    Try to create an output directory if one doesn't exist.

    Throws an exception if the directory cannot be created.
    Returns quietly if the directory already exists.

    Args:
        dir_ : name of directory

    Returns:
        str: path to directory (see :mod:`os.path`)
    """
    try:
        if not exists(dir_):
            mkdir(dir_)
        else:
            return dir_
    except OSError:
        print(f'Cannot create directory "{realpath(dir_)}"')
        raise
    except Exception:
        print(Exception)
        raise
    return dir_


def export_results(
    results_dir: PathLike, filename: PathLike, raw_dict: Dict, suffix: str = ""
) -> None:
    """
    Export results dictionary to JSON file.

    Tries to ensure all dictionary entries are
    serializable by running `latex`
    on keys and converting values to floats.

    Args:
        results_dir:
            name of output directory

        filename:
            name of output JSON file

        suffix:
            to append to filename prior to addition of '.json' extension

        raw_dict:
            dictionary of results, possibly requiring conversion
            from latex form such that serialization into a JSON file
            is possible
    """
    serializable_dict = {}
    for item in raw_dict.items():
        serializable_dict.update({latex(item[0]): float(item[1])})
    results_path_ = [str(results_dir)] + [
        str(filename) + "_" + suffix + ".json"
    ]
    with open(join(*results_path_), "w", encoding="latin-1") as fp:
        logging.info(join(*results_path_))
        dump(serializable_dict, fp, separators=(", \n", ": "))


def export_plots(
    fig_dict: Dict,
    results_dir: PathLike,
    file_types: Union[List[str], Tuple[str], str] = "pdf",
    suffix: str = "",
    dpi: Optional[int] = None,
) -> None:
    """
    Export plots to PDF or other format files.

    Args:
        fig_dict:
            dictionary of figures
        results_dir:
            name of output directory
        file_types:
            file format (or list of file formats)
        suffix:
            filename suffix
    """
    results_path: PathLike = realpath(results_dir)
    logging.info(
        "gmplib.save.export_plots:\n   " + f'Writing to dir: "{results_path}"'
    )
    file_types_: List[str] = (
        file_types if isinstance(file_types, list) else [str(file_types)]
    )
    for file_type in file_types_:
        # logging.info(f'Image file type: "{file_type}"')
        for fig_dict_item in list(fig_dict.items()):
            export_plot(
                *fig_dict_item,
                results_path,
                file_type=file_type,
                suffix=suffix,
                dpi=dpi,
            )


def export_plot(
    fig_name: str,
    fig: figure,
    results_dir: PathLike,
    file_type: str = "pdf",
    suffix: str = "",
    dpi: Optional[int] = None,
) -> None:
    """
    Export plot to PDF or other format file.

    Args:
        fig_name:
            name to be used for file (extension auto-appended)
        fig:
            figure object
        results_dir:
            name of output directory
        file_type:
            file format
        suffix:
            filename suffix
    """
    fig_name_ = f"{fig_name}{suffix}.{file_type.lower()}"
    try:
        # logging.info(f'dpi={dpi}')
        fig.savefig(
            join(results_dir, fig_name_),
            bbox_inches="tight",
            pad_inches=0.05,
            dpi=dpi,
            format=file_type,
        )
        logging.info(f'gmplib.save.export_plot: Exported "{fig_name_}"')
    except OSError:
        logging.info(
            f'gmplib.save.export_plot: Failed to export figure "{fig_name_}"'
        )
        raise
    # except:
    #     raise
