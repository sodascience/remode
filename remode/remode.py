"""Recursive Mode Detection (ReMoDe) for ordinal data."""

from collections import Counter
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy.stats import binomtest, fisher_exact


def perform_fisher_test(
    x: np.ndarray, candidate: int, left_min: int, right_min: int
) -> Tuple[float, float]:
    """
    Perform Fisher's exact test on both sides of a candidate maximum to test if it is a true local maximum.

    Parameters
    ----------
    x : np.ndarray
        The input data array.
    candidate : int
        The index of the candidate maximum.
    left_min : int
        The index of the minimum value on the left side of the candidate maximum.
    right_min : int
        The index of the minimum value on the right side of the candidate maximum.

    Returns
    -------
    Tuple[float, float]
        The p-values of the Fisher's exact test for the left and right sides of the candidate maximum.
    """
    left_matrix = np.array(
        [
            [x[candidate], np.sum(x) - x[candidate]],
            [x[left_min], np.sum(x) - x[left_min]],
        ]
    )
    right_matrix = np.array(
        [
            [x[candidate], np.sum(x) - x[candidate]],
            [x[right_min], np.sum(x) - x[right_min]],
        ]
    )
    p_left = fisher_exact(left_matrix, "greater")[1]
    p_right = fisher_exact(right_matrix, "greater")[1]
    return p_left, p_right


def perform_binomial_test(
    x: np.ndarray, candidate: int, left_min: int, right_min: int
) -> Tuple[float, float]:
    """
    Perform binomial tests on both sides of a candidate maximum to test if it is a true local maximum.

    Parameters
    ----------
    x : np.ndarray
        The input data array.
    candidate : int
        The index of the candidate maximum.
    left_min : int
        The index of the minimum value on the left side of the candidate maximum.
    right_min : int
        The index of the minimum value on the right side of the candidate maximum.

    Returns
    -------
    Tuple[float, float]
        The p-values of the binomial tests for the left and right sides of the candidate maximum.
    """
    n_left = x[candidate] + x[left_min]
    n_right = x[candidate] + x[right_min]
    p_left = binomtest(x[candidate], n_left, alternative="greater").pvalue
    p_right = binomtest(x[candidate], n_right, alternative="greater").pvalue
    return p_left, p_right


class ReMoDe:
    """
    ReMoDe (Robust Mode Detection) is a class for detecting modes in a dataset.

    The class uses statistical tests to identify local maxima in the dataset that are statistically significant.
    The identified modes can then be used for further analysis.

    Attributes
    ----------
    alpha : float
        The significance level for the statistical tests. Default is 0.05.
    alpha_correction : str or function
        The method for correcting the significance level for multiple comparisons. Options are "max_modes" and "none". Default is "none".
    statistical_test : function
        The statistical test to use for identifying local maxima. Options are `perform_fisher_test` and `perform_binomial_test`. Default is `perform_fisher_test`.

    Methods
    -------
    format_data(xt: np.ndarray) -> np.ndarray:
        Formats the input data for mode detection.
    fit(xt: np.ndarray) -> Dict[str, Union[int, np.ndarray]]:
        Fits the model to the input data and returns the detected modes.
    evaluate_stability(iterations: int, percentage_steps: int) -> Dict[str, Any]:
        Evaluates the stability of the detected modes using the jackknife resampling method.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        alpha_correction: Union[Literal["max_modes", "none"], Callable] = "none",
        statistical_test: Callable = perform_fisher_test,
    ):
        self.alpha = alpha

        if isinstance(alpha_correction, str):
            if alpha_correction.lower() == "none":
                self._create_alpha_correction = lambda length, alpha: alpha
            elif alpha_correction.lower() == "max_modes":
                self._create_alpha_correction = lambda length, alpha: alpha / (
                    np.floor((length + 1) / 2)
                )
        else:
            # Test if the provided function is valid (has one argument)
            if not callable(alpha_correction):
                raise ValueError(
                    "The alpha_correction argument must be a function or one of 'max_modes' or 'none'."
                )
            elif len(alpha_correction.__code__.co_varnames) != 2:
                raise ValueError(
                    "The alpha_correction function must take two arguments (the legnth of the bins and the alpha level)."
                )
            self._create_alpha_correction = alpha_correction

        self.alpha_cor: Optional[float] = None
        self.statistical_test = statistical_test
        self.modes: np.ndarray = np.array([])
        self.xt: np.ndarray = np.array([])
        self.levels: np.ndarray = np.array([])

    def format_data(
        self, xt: Union[np.ndarray, list], levels: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Formats the input data for mode detection.

        Parameters
        ----------
        xt : np.ndarray
            The input data array.
        levels : np.ndarray, optional
            The levels at which to find the modes. If not provided, defaults to the range of the length of the input data.

        Returns
        -------
        np.ndarray
            The formatted data array.
        """
        if levels is None:
            # Ensure xt is a numpy array for processing
            levels = np.unique(xt)

        levels_h = np.concatenate([levels - 0.01, [levels[-1] + 0.01]])

        return np.histogram(xt, bins=levels_h)[0]

    def _find_maxima(self, xt: np.ndarray) -> np.ndarray:
        """
        Finds the local maxima in the input data.

        Parameters
        ----------
        xt : np.ndarray
             The input data array.

        Returns
        -------
        np.ndarray
            The indices of the local maxima in the input data.
        """
        if len(xt) < 3:
            return np.array([], dtype=int)

        result = []
        candidate = np.argmax(xt)
        if candidate != 0 and candidate != len(xt) - 1:
            left_min = np.argmin(xt[:candidate])
            right_min = np.argmin(xt[candidate:]) + candidate
            p_left, p_right = self.statistical_test(xt, candidate, left_min, right_min)
            if p_left < self.alpha_cor and p_right < self.alpha_cor:
                result.append(candidate)
        result.extend(self._find_maxima(xt[:candidate]))
        result.extend(self._find_maxima(xt[candidate + 1 :]) + candidate + 1)
        return np.unique(result)

    def fit(
        self, xt: np.ndarray, set_data: bool = True, levels: np.ndarray = np.array([])
    ) -> Dict[str, Any]:
        """
        Fits the model to the input data and returns the detected modes.

        This method first applies an alpha correction to the input data, then finds the local maxima (modes) in the data.
        If `levels` is not provided, it defaults to the range of the length of the input data.
        If `set_data` is True, the input data is stored in the `xt` attribute of the class instance.

        Parameters
        ----------
        xt : np.ndarray
            The input data array.
        set_data : bool, optional
            Whether to store the input data in the `xt` attribute of the class instance. Default is True.
        levels : np.ndarray, optional
            The levels at which to find the modes. If not provided, defaults to the range of the length of the input data.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the detected modes and their properties. The keys of the dictionary are the indices of the modes, and the values are the properties of the modes.
        """
        self.alpha_cor = self._create_alpha_correction(len(xt), self.alpha)
        xt_padded: np.ndarray = np.concatenate((np.array([0]), xt, np.array([0])))
        modes = self._find_maxima(xt_padded)
        modes -= 1

        if len(levels) == 0:
            self.levels = np.arange(len(xt))
        else:
            self.levels = levels

        if set_data:
            self.xt = xt
            self.modes = modes

        return {
            "nr_of_modes": len(modes),
            "modes": modes,
            "xt": xt,
            "alpha_after_correction": self.alpha_cor,
        }

    def plot_maxima(
        self, ax: Optional[Axes] = None, format_plot: Optional[bool] = True
    ) -> None:
        """
        Formats the input data for mode detection.

        Parameters
        ----------
        xt : np.ndarray
            The input data array.
        format_plot : bool, optional
            Whether to format the plot. Default is True.

        Returns
        -------
        np.ndarray
            The formatted data array.
        """

        if len(self.xt) == 0:
            raise ValueError("Please fit the model first before plotting the maxima.")

        if ax is None:
            _, ax = plt.subplots()

        ax.bar(self.levels, self.xt, color="lightgrey", zorder=9, label=None)
        if len(self.modes) > 0:
            ax.bar(
                np.take(self.levels, self.modes),
                np.take(self.xt, self.modes),
                color="navy",
                zorder=10,
                alpha=0.5,
                label="Modes",
            )

        if format_plot:
            ax.set_xticks(self.levels)
            self._format_plot(ax)

    def _format_plot(
        self,
        ax,
        xlabel="Category",
        ylabel="Frequency",
        title="Modes detected",
        legend=True,
    ):
        """
        Formats the plot with the given title and standard settings.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The Axes object to format.
        xlabel : str
            The label to set for the x-axis.
        ylabel : str
            The label to set for the y-axis.
        title : str
            The title to set for the plot.
        legend : bool
            Whether to show the legend. Default is True.

        Returns
        -------
        None
        """
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_title(title, loc="left")
        if legend:
            ax.legend()

    def _jackknife(self, percentage: float) -> np.ndarray:
        """
        Performs the jackknife resampling method on the input data.

        Parameters
        ----------
        percentage : float
            The percentage of data to remove in each resampling step.

        Returns
        -------
        np.ndarray
            The resampled data array.
        """
        # Create data from count
        x = np.repeat(np.arange(len(self.xt)), self.xt)

        # Remove 20% of the data
        n_delete = int(percentage * len(x) / 100)

        if n_delete > 0:
            x_remove = np.random.choice(x, n_delete, replace=False)
            remove_count = np.bincount(x_remove, minlength=len(self.xt))

            return np.bincount(x, minlength=len(self.xt)) - remove_count

        return np.bincount(x, minlength=len(self.xt))

    def evaluate_stability(
            self, iterations: int = 100, percentage_steps: int = 10, plot: bool = True
        ) -> Dict[str, Any]:
        """
        Evaluates the stability of the detected modes using the jackknife resampling method.

        Parameters
        ----------
        iterations : int
            The number of iterations to perform for the jackknife resampling.
        percentage_steps : int
            The number of percentage steps to remove in the jackknife resampling.
        plot : bool
            Whether to plot the stability analysis results.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the results of the stability evaluation.
        """
        if len(self.modes) == 0:
            raise ValueError(
                "It appears that either you did not yet apply ReMoDe or that no modes were found. Stability analyses can only be performed on modes detected by ReMoDe."
            )


        perc_range = np.linspace(0, 100, percentage_steps)
        modes = pd.DataFrame(
            {
                "perc": perc_range,
                "mean_modality": np.nan,
                "most_freq_modality": np.nan,
                "majority_result": False,
            }
        )

        modes.at[0, "mean_modality"] = len(self.modes)
        modes.at[0, "most_freq_modality"] = len(self.modes)
        modes.at[0, "majority_result"] = True

        # Initialize matrix to store data counts of mode locations
        modes_locations = np.zeros((percentage_steps + 1, len(self.xt)))
        modes_locations[0, self.modes] = iterations

        for i in range(1, len(perc_range)):
            m = np.zeros(iterations)
            for j in range(iterations):
                xt_jackknifed = self._jackknife(modes.at[i, "perc"])
                r = self.fit(xt_jackknifed, set_data=False)
                m[j] = r["nr_of_modes"]
                # Update mode location matrix
                for mode in r["modes"]:
                    modes_locations[i, mode] += 1

            modes.at[i, "mean_modality"] = np.mean(m)
            modes.at[i, "most_freq_modality"] = Counter(m).most_common(1)[0][0]
            modes.at[i, "majority_result"] = (
                np.mean(m == modes.at[i, "most_freq_modality"]) >= 0.5
            )


        modes.loc[len(perc_range) - 1, ["mean_modality", "most_freq_modality", "majority_result"]] = [0, 0, False]
        modes_locations[len(perc_range) - 1, :] = 0

        stable_until = modes.loc[
            (modes["majority_result"] == 1) & (modes["most_freq_modality"] == modes.at[0, "most_freq_modality"]),
            "perc"
        ].max()

        # Calculate the stability of the location of detected modes
        stability_location = np.apply_along_axis(
            lambda x: (np.argmax(x > (iterations / 2)) - 1) / len(perc_range),
            axis=0,
            arr=modes_locations
        )

        stability_location = stability_location[stability_location > 0]
        # Ensure compatibility in dimensions
        if len(stability_location) > 0:
            stability_location = np.column_stack((sorted(self.modes), stability_location))
            stability_location_df = pd.DataFrame(stability_location, columns=["Mode location", "Stability estimate"])
        else:
            stability_location_df = pd.DataFrame(columns=["Mode location", "Stability estimate"])

        if plot:
            plt.figure(figsize=(12, 4))
            gs = gridspec.GridSpec(1, 2)  # 1 row, 2 columns

            ax1 = plt.subplot(gs[0, 0])  # left subplot spans first two columns
            self.plot_maxima(ax1)

            ax2 = plt.subplot(gs[0, 1])  # right subplot in the third column
            plt.step(
                modes["perc"],
                modes["mean_modality"],
                where="mid",
                label="Mean Modality",
                color="navy",
            )
            plt.step(
                modes["perc"],
                modes["most_freq_modality"],
                where="mid",
                linestyle="--",
                label="Most Frequent Modality",
                color="cornflowerblue",
            )
            self._format_plot(
                ax2,
                xlabel="Percentage of Data Removed",
                ylabel="Modality",
                title=f"Modes: {len(self.modes)}, stability {stable_until}%",
                legend=True,
            )
            plt.tight_layout()

        return {
            "num_mode_stability": modes,
            "stable_until": stable_until,
            "location_stability": stability_location_df
        }
