# ReMoDe: a Python library for efficient mode detection in ordinal data distributions.

`ReMoDe` (Recursive Mode Detection) is a Python library designed for the robust detection of modes in ordinal data distributions. It uses statistical tests, including Fisher's exact test and binomial tests, to determine if a given maximum in a data distribution is a true local maximum.

### Features
- Mode Detection: Identifies all potential local maxima in the dataset.
- Statistical Tests: Implements Fisher's exact test and binomial tests to validate modes.
- Data Formatting: Converts raw data into histogram format for analysis.
- Stability Analysis: Includes functionality to assess the stability of detected modes using jackknife resampling.
- Visualization: Provides methods to plot the histogram of data along with identified modes.

### Installation

```bash
pip install remode
```

### Usage

Here is a simple example of how to use the ReMode library:

```python
from remode import ReMoDe

# Sample data (histogram counts)
xt_count = [8, 20, 5, 2, 6, 2, 30]

# Create an instance of ReMoDe
detector = ReMoDe()

# Fit model
results = detector.fit(xt_count)

# Plot the results
detector.plot_maxima()

# Perform stability analysis
stability_info = detector.evaluate_stability(percentage_steps=50)

```

See also the tutorial [here](https://github.com/sodascience/remode/blob/main/tutorial.ipynb).

### Citation

Please cite the following paper:
```
TBA
```
and the following software
```
Garcia-Bernardo, J., Hoffstadt, M., & van der Maas, H. L. J. (2025). ReMoDe: a Python library for efficient mode detection in ordinal data distributions. Zenodo. https://doi.org/10.5281/zenodo.15366121
```

### Contributing

Contributions are what make the open source community an amazing place
to learn, inspire, and create. Any contributions you make are **greatly
appreciated**.

Please refer to the
[CONTRIBUTING](https://github.com/sodascience/remode/blob/main/CONTRIBUTING.md)
file for more information on issues and pull requests.


### License

This project is licensed under the GNU GPLv3. This allows you to do almost anything they want with this project, except distributing closed source versions. 


## Contact

This project is a port of the R version of [`ReMoDe`](https://github.com/hvdmaas/remode). It is maintained by the [ODISSEI Social Data
Science (SoDa)](https://odissei-data.nl/nl/soda/) team.

<img src="soda_logo.png" alt="SoDa logo" width="250px"/>

Do you have questions, suggestions, or remarks? File an issue in the issue
tracker!
