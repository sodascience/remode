from collections import Counter
from typing import Any, Dict, Optional, Tuple, Union

from matplotlib.axes import Axes
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact


def fisher_test(x: np.ndarray, candidate: int, left_min: int, right_min: int) -> Tuple[float, float]:
    """
    Perform Fisher's exact test on both sides of a candidate maximum to test if it is a true local maximum.
    """
    left_matrix = np.array([[x[candidate], np.sum(x) - x[candidate]], [x[left_min], np.sum(x) - x[left_min]]])
    right_matrix = np.array([[x[candidate], np.sum(x) - x[candidate]], [x[right_min], np.sum(x) - x[right_min]]])
    p_left = fisher_exact(left_matrix, 'greater')[1]
    p_right = fisher_exact(right_matrix, 'greater')[1]
    return p_left, p_right

def binomial_test(x: np.ndarray, candidate: int, left_min: int, right_min: int) -> Tuple[float, float]:
    """
    Perform binomial tests on both sides of a candidate maximum to test if it is a true local maximum.
    """
    n_left = x[candidate] + x[left_min]
    n_right = x[candidate] + x[right_min]
    p_left = binomtest(x[candidate], n_left, alternative='greater').pvalue
    p_right = binomtest(x[candidate], n_right, alternative='greater').pvalue
    return p_left, p_right

class ReMode:
    def __init__(self, alpha: float = 0.05, alpha_correction: int = 2, statistical_test: Any = fisher_test):
        self.alpha = alpha
        self.alpha_correction = alpha_correction
        self.alpha_cor = None
        self.statistical_test = statistical_test
        self.modes = None
        self.xt = None
        self.levels = None

    def format_data(self, xt: Union[np.ndarray, list], levels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Formats raw data into a histogram based on provided levels.
        
        Args:
            xt: Raw data to be formatted, which can be a list or numpy array.
            levels: Bin levels for the histogram. If None, unique elements of xt are used.
            
        Returns:
            Formatted data as histogram counts.
        """
        if levels is None:
            # Ensure xt is a numpy array for processing
            levels = np.unique(xt)
        
        levels_h = np.concatenate([levels - 0.01, [levels[-1] + 0.01]])
    
        return np.histogram(xt, bins=levels_h)[0]

    def _find_maxima(self, x: np.ndarray) -> np.ndarray:
        if len(x) < 3:
            return np.array([], dtype=int)
    
        
        result = []
        candidate = np.argmax(x)
        if candidate != 0 and candidate != len(x) - 1:            
            left_min = np.argmin(x[:candidate])
            right_min = np.argmin(x[candidate:]) + candidate
            p_left, p_right = self.statistical_test(x, candidate, left_min, right_min)
            
            if p_left < self.alpha_cor and p_right < self.alpha_cor:
                result.append(candidate)
        result.extend(self._find_maxima(x[:candidate]))
        result.extend(self._find_maxima(x[candidate + 1:]) + candidate + 1)
        return np.unique(result)

    def fit(self, xt: np.ndarray, set_data: bool = True, levels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        self.alpha_cor = self._get_corrected_alpha(len(xt))
        xt_padded = np.concatenate(([0], xt, [0]))
        modes = self._find_maxima(xt_padded)
        modes -= 1
        
        if levels is None:
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
            "alpha": self.alpha_cor,
            "alpha_correction": self.alpha_correction
        }

    def _get_corrected_alpha(self, length: int) -> float:
        return {
            1: self.alpha / (length - 1),
            2: self.alpha / (np.floor((length + 1) / 2)),
            3: 2 * self.alpha / (0.5 * length * (length - 1)),
            4: self.alpha / 3,
            5: self.alpha,
            6: self.alpha / np.sqrt(length)
        }.get(self.alpha_correction)


    def plot_maxima(self, ax: Optional[Axes] = None, format_plot: Optional[bool] = True) -> None:
        """
        Plots the data and highlights the modes.
        
        Args:
            ax: Optional Matplotlib Axes object on which to plot. If None, a new figure and axes object are created.
        """

        if self.xt is None:
            raise ValueError("Please fit the model first before plotting the maxima.")
        
        if ax is None:
            fig, ax = plt.subplots()

        ax.bar(self.levels, self.xt, color='lightgrey', zorder=9, label=None)
        if len(self.modes) > 0:
            ax.bar(np.take(self.levels, self.modes), np.take(self.xt, self.modes), color='navy', zorder=10, alpha=0.5, label='Modes')
        
        if format_plot:
            ax.set_xticks(self.levels)
            self._format_plot(ax)

    def _format_plot(self, ax, xlabel="Category", ylabel="Frequency", title="Modes detected", legend=True):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_title(title, loc="left")
        if legend:
            ax.legend()



    def jackknife(self, percentage: float) -> np.ndarray:
        """
        Performs the jackknife resampling method by randomly deleting a percentage of data.
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
    
    ##Continue here
    def remode_robustness(self, iterations: int = 100, percentage_steps: int = 10) -> Dict[str, Any]:
        """
        Analyze the robustness of the mode estimation by repeatedly applying jackknife resampling.
        """
        if self.modes is None:
            raise ValueError("Please fit the model first before performing robustness analysis.")
        
        perc_range = np.linspace(0, 100, percentage_steps)
        modes = pd.DataFrame({
            'perc': perc_range,
            'mean_modality': np.nan,
            'most_freq_modality': np.nan,
            'majority_result': np.nan
        })
        
        modes.at[0, 'mean_modality'] = len(self.modes)
        modes.at[0, 'most_freq_modality'] = len(self.modes)
        modes.at[0, 'majority_result'] = True

        for i in range(1, len(perc_range)):
            m = np.zeros(iterations)
            for j in range(iterations):
                xt_jackknifed = self.jackknife(modes.at[i, 'perc'])
                m[j] = self.fit(xt_jackknifed, set_data=False)['nr_of_modes']

            modes.at[i, 'mean_modality'] = np.mean(m)
            modes.at[i, 'most_freq_modality'] = Counter(m).most_common(1)[0][0]
            modes.at[i, 'majority_result'] = np.mean(m == modes.at[i, 'most_freq_modality']) >= 0.5

        modes.loc[len(perc_range)-1, ['mean_modality', 'most_freq_modality', 'majority_result']] = [0, 0, False]
        robust_until = modes['perc'][modes['majority_result'] == 1].max()


        fig = plt.figure(figsize=(12, 4))
        gs = gridspec.GridSpec(1, 3)  # 1 row, 3 columns

        ax1 = plt.subplot(gs[0, :1])  # left subplot spans first two columns
        self.plot_maxima(ax1)

        ax2 = plt.subplot(gs[0, 1:])  # right subplot in the third column
        plt.step(modes['perc'], modes['mean_modality'], where='mid', label='Mean Modality', color='navy')
        plt.step(modes['perc'], modes['most_freq_modality'], where='mid', linestyle='--', label='Most Frequent Modality', color='cornflowerblue')
        self._format_plot(ax2, xlabel='Percentage of Data Removed', ylabel='Modality', title='Robustness Analysis', legend=True)
        

        plt.tight_layout()
        

        return {
            "jacknife_df": modes,
            "robust_until": robust_until
        }
