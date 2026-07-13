from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

BG = "#fbfaf7"
GRID = "#ded8cf"
TEXT = "#2f2d2a"

FIGSIZE_TRAIN = (6.8, 3.65)
FIGSIZE_SEARCH = (8.2, 4.65)
FIGSIZE_TRAJECTORY_ANNOTATED = (10.6, 6.35)
FIGSIZE_ABLATION = (6.9, 3.85)


RC = {
    "font.family": "DejaVu Sans",
    "font.size": 8.2,
    "axes.labelsize": 8.6,
    "axes.titlesize": 9.0,
    "axes.titleweight": "semibold",
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.6,
    "legend.fontsize": 7.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 160,
    "savefig.dpi": 200,
    "axes.facecolor": BG,
    "figure.facecolor": BG,
    "axes.edgecolor": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "text.color": TEXT,
    "grid.color": GRID,
    "grid.linewidth": 0.65,
    "grid.alpha": 0.55,
    "lines.linewidth": 1.9,
    "patch.linewidth": 0.8,
}


@contextmanager
def paper_style(
    *,
    figsize: tuple[float, float],
    suppress_titles: bool = True,
    suppress_free_text: bool = False,
    text_scale: float = 1.0,
    annotate_scale: float = 1.0,
    extra_save_formats: tuple[str, ...] = (),
) -> Iterator[None]:
    old_rc = plt.rcParams.copy()
    old_subplots = plt.subplots
    old_figure = plt.figure
    old_set_title = Axes.set_title
    old_set_xlabel = Axes.set_xlabel
    old_set_ylabel = Axes.set_ylabel
    old_legend = Axes.legend
    old_text = Axes.text
    old_annotate = Axes.annotate
    old_figure_savefig = Figure.savefig
    old_figure_text = Figure.text
    old_suptitle = Figure.suptitle
    old_plt_title = plt.title

    def subplots_wrapper(*args, **kwargs):
        kwargs["figsize"] = figsize
        return old_subplots(*args, **kwargs)

    def figure_wrapper(*args, **kwargs):
        kwargs["figsize"] = figsize
        return old_figure(*args, **kwargs)

    def set_title_wrapper(self, label, *args, **kwargs):
        if suppress_titles:
            return old_set_title(self, "", *args, **kwargs)
        kwargs["fontsize"] = 9.0
        return old_set_title(self, label, *args, **kwargs)

    def set_xlabel_wrapper(self, xlabel, *args, **kwargs):
        kwargs["fontsize"] = 8.6
        kwargs.setdefault("labelpad", 4)
        return old_set_xlabel(self, xlabel, *args, **kwargs)

    def set_ylabel_wrapper(self, ylabel, *args, **kwargs):
        kwargs["fontsize"] = 8.6
        kwargs.setdefault("labelpad", 4)
        return old_set_ylabel(self, ylabel, *args, **kwargs)

    def legend_wrapper(self, *args, **kwargs):
        kwargs["fontsize"] = 6.9
        kwargs.setdefault("title_fontsize", 7.2)
        legend = old_legend(self, *args, **kwargs)
        legend.get_frame().set_linewidth(0.65)
        legend.get_frame().set_alpha(0.94)
        return legend

    def title_wrapper(label, *args, **kwargs):
        if suppress_titles:
            label = ""
        return old_plt_title(label, *args, **kwargs)

    def suptitle_wrapper(self, t, *args, **kwargs):
        if suppress_titles:
            t = ""
        return old_suptitle(self, t, *args, **kwargs)

    def text_wrapper(self, *args, **kwargs):
        if suppress_free_text:
            if len(args) >= 3 and str(args[2]).strip().isdigit():
                if "fontsize" in kwargs and isinstance(kwargs["fontsize"], (int, float)):
                    kwargs["fontsize"] = float(kwargs["fontsize"]) * float(text_scale)
                return old_text(self, *args, **kwargs)
            return old_text(self, 0, 0, "", alpha=0)
        if "fontsize" in kwargs and isinstance(kwargs["fontsize"], (int, float)):
            kwargs["fontsize"] = float(kwargs["fontsize"]) * float(text_scale)
        return old_text(self, *args, **kwargs)

    def annotate_wrapper(self, *args, **kwargs):
        if suppress_free_text:
            return old_annotate(self, "", xy=(0, 0), alpha=0)
        if "fontsize" in kwargs and isinstance(kwargs["fontsize"], (int, float)):
            kwargs["fontsize"] = float(kwargs["fontsize"]) * float(annotate_scale)
        return old_annotate(self, *args, **kwargs)

    def figure_text_wrapper(self, *args, **kwargs):
        if suppress_free_text:
            return old_figure_text(self, 0, 0, "", alpha=0)
        return old_figure_text(self, *args, **kwargs)

    def figure_savefig_wrapper(self, fname, *args, **kwargs):
        result = old_figure_savefig(self, fname, *args, **kwargs)
        if extra_save_formats and isinstance(fname, (str, Path)):
            path = Path(fname)
            if path.suffix.lower() == ".png":
                for fmt in extra_save_formats:
                    suffix = "." + fmt.lower().lstrip(".")
                    if suffix == ".png":
                        continue
                    old_figure_savefig(self, path.with_suffix(suffix), *args, **kwargs)
        return result

    try:
        plt.rcParams.update(RC)
        plt.subplots = subplots_wrapper
        plt.figure = figure_wrapper
        Axes.set_title = set_title_wrapper
        Axes.set_xlabel = set_xlabel_wrapper
        Axes.set_ylabel = set_ylabel_wrapper
        Axes.legend = legend_wrapper
        Axes.text = text_wrapper
        Axes.annotate = annotate_wrapper
        Figure.savefig = figure_savefig_wrapper
        Figure.text = figure_text_wrapper
        Figure.suptitle = suptitle_wrapper
        plt.title = title_wrapper
        yield
    finally:
        plt.rcParams.update(old_rc)
        plt.subplots = old_subplots
        plt.figure = old_figure
        Axes.set_title = old_set_title
        Axes.set_xlabel = old_set_xlabel
        Axes.set_ylabel = old_set_ylabel
        Axes.legend = old_legend
        Axes.text = old_text
        Axes.annotate = old_annotate
        Figure.savefig = old_figure_savefig
        Figure.text = old_figure_text
        Figure.suptitle = old_suptitle
        plt.title = old_plt_title
