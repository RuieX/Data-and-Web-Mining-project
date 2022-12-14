import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import project.src.feat_eng as fe


def plot_feature_distribution(data, var, subplot_size: (int, int), width: int = 5, cat: bool = False):
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

    return fig, axs


def plot_correlation(data, column):
    cor_matrix = data[column].corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones_like(cor_matrix, dtype=bool), k=1))
    plt.figure(figsize=(12, 8))
    sns.heatmap(data=upper_tri)
    plt.show()
    plt.gcf().clear()


def missing_values_plot(data, subplot_size: (int, int), width: int = 7, **barplot_kwargs):

    # Get dataframe containing missing values information
    features_col = data.T.columns
    pct_col = data['Missing pct']

    # Create plotting grid
    n_features = len(data.T.columns)
    fig, axs = _get_plotting_grid(width, n_features, subplot_size, style="whitegrid")

    # Create a plot for each grid square
    plot_row = 0
    for i, feature in enumerate(data.T.columns):
        height = axs.shape[1]
        plot_col = i % height

        # Move to next row when all cols have been plotted
        if i != 0 and plot_col == 0:
            plot_row += 1

        plot_onto = axs[plot_row, plot_col]

        # Mask used to extract values belonging to the current feature
        # from the missing info dataframe
        current_feature_mask = features_col == feature

        # Get the current feature missing information
        missing_info = pct_col[current_feature_mask]
        plot_onto.set_ylim([0, 100])

        # Show tick every 10%
        plot_onto.set_yticks(list(range(0, 101, 10)))

        sns.barplot(ax=plot_onto, x=[feature], y=missing_info, **barplot_kwargs)


def bivariate_feature_plot(data: pd.DataFrame, y_var: (str, pd.Series),
                           subplot_size: (int, int), mode: str = "scatter",
                           width: int = 3, percentile_range: (float, float) = (0, 100),
                           show_legend: bool = True,
                           hexbin_kwargs: dict[str, object] = None,
                           scatter_kwargs: dict[str, object] = None) -> (plt.Figure, np.ndarray):
    # Create grid
    n_features = len(data.columns)
    fig, axs = _get_plotting_grid(width, n_features, subplot_size)

    # Create a plot for each grid square
    plot_row = 0
    for i, col in enumerate(data.columns):
        height = axs.shape[1]
        plot_col = i % height

        # Move to next row when all cols have been plotted
        if i != 0 and plot_col == 0:
            plot_row += 1

        feature = data[col]
        y_name, y_data = y_var

        # Get the data withing the specified percentile range
        lower_q = percentile_range[0] / 100
        upper_q = percentile_range[1] / 100

        quantile_range_mask = (feature >= feature.quantile(lower_q)) & (feature <= feature.quantile(upper_q)) \
                              & (y_data >= y_data.quantile(lower_q)) & (y_data <= y_data.quantile(upper_q))
        x_ranged = feature[quantile_range_mask]
        y_ranged = y_data[quantile_range_mask]

        # Set x and y labels to feature and y_var names
        plot_onto = axs[plot_row, plot_col]
        plot_onto.set_xlabel(col)
        plot_onto.set_ylabel(y_name)

        if mode == "hexbin":
            hexbin = plot_onto.hexbin(x=x_ranged.values, y=y_ranged.values,
                                      **hexbin_kwargs)
            if show_legend:
                cb = fig.colorbar(hexbin, ax=plot_onto)
                cb.set_label('counts')
        else:
            # Select the data from the original dataframe in order to keep the other columns:
            # this  way, seaborn kwargs that refer to such columns (e.g. hue, size) can be passed
            scatter_data = data.copy()
            scatter_data = scatter_data[scatter_data[col].isin(x_ranged.values)]
            scatter_data[y_name] = y_data
            scatter_data = scatter_data[scatter_data[y_name].isin(y_ranged.values)]

            sns.scatterplot(ax=plot_onto, data=scatter_data, x=col, y=y_name,
                            **scatter_kwargs)

            if not show_legend:
                legend = plot_onto.get_legend()

                if legend is not None:
                    legend.remove()

    return fig, axs


def feature_target_scatter_plot(data: fe.TrainTestSplit):
    subplot_width = 8
    subplot_height = 6
    plots_width = 3
    bivariate_feature_plot(data=data.x_train,
                           y_var=("target", pd.Series(data.y_train)),
                           mode="scatter", show_legend=False,
                           subplot_size=(subplot_width, subplot_height),
                           width=plots_width,
                           scatter_kwargs={})


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
