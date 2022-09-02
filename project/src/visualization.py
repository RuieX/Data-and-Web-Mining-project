import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def feature_distribution(data, var, subplot_size: (int, int), width: int = 5, cat: bool = False):
    # Create plotting grid
    n_features = len(var)
    fig, axs = _get_plotting_grid(width, n_features, subplot_size, style="whitegrid")

    plot_row = 0
    for i, col in enumerate(var):
        height = axs.shape[1]
        plot_col = i % height

        # Move to next row when all cols have been plotted
        if i != 0 and plot_col == 0:
            plot_row += 1

        feature_to_plot = data[col]
        plot_onto = axs[plot_row, plot_col]

        # Check if the feature is numerical or categorical
        n_uniques = len(feature_to_plot.unique())
        if not cat:
            sns.boxplot(ax=plot_onto, x=feature_to_plot, color="#ffa500")
        else:
            # reset index is done because it makes the unique values
            # accessible in the new "index" column. This way, you
            # have a plot where x is the unique value ("index") and
            # y is the count (same name of the column)
            val_counts = feature_to_plot.value_counts(normalize=True).reset_index()
            plot_ax = sns.barplot(ax=plot_onto, data=val_counts, x="index", y=col, palette="pastel")
            plot_ax.set(xlabel=col, ylabel="frequency")


def _get_plotting_grid(width: int, tot_cells: int, subplot_size: (int, int),
                       style: str = "ticks", **subplots_kwargs) -> (plt.Figure, np.ndarray):
    """
    Returns a plot grid based on the provided parameters.

    :param width: width (in plots) of the grid
    :param tot_cells: total number of cells (plots) of the grid
    :param subplot_size: dimension of each subplot in the grid
    :param style: seaborn style of the plots
    :param subplots_kwargs: additional kwargs passed to the underlying pyplot.subplots call
    :return: fig, axs
    """
    sns.set_style(style=style)

    # Calculate dimensions of the grid
    height = tot_cells // width
    if width * height < tot_cells or height == 0:
        height += 1

    fig_width = width * subplot_size[0]
    fig_height = height * subplot_size[1]

    fig, axs = plt.subplots(ncols=width, nrows=height, figsize=(fig_width, fig_height), **subplots_kwargs)
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)  # Else fig.suptitle overlaps with the subplots

    return fig, axs