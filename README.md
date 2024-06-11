# ReMoDe

`ReMoDe` is a Python library designed for the robust detection of modes in data distributions. It uses statistical tests, including Fisher's exact test and binomial tests, to determine if a given maximum in a data distribution is a true local maximum.

### Features
- Mode Detection: Identifies all potential local maxima in the dataset.
- Statistical Tests: Implements Fisher's exact test and binomial tests to validate modes.
- Data Formatting: Converts raw data into histogram format for analysis.
- Robustness Analysis: Includes functionality to assess the robustness of detected modes using jackknife resampling.
- Visualization: Provides methods to plot the histogram of data along with identified modes.

### Installation

Clone this repository to your local machine:

bash

git clone https://github.com/yourusername/remode.git

Navigate to the remode directory and install the required dependencies:

bash

cd remode
pip install -r requirements.txt

### Usage

Here is a simple example of how to use the ReMode library:

```python
from remode import ReMode

# Sample data (histogram counts)
xt_count = [20, 21, 20, 18, 35, 24, 21, 28, 35]

# Create an instance of ReMoDe
detector = ReMode()

# Fit model
results = detector.fit(data)

# Plot the results
detector.plot_maxima()

# Perform robustness analysis
robustness_info = detector.remode_robustness(iterations=100, percentage_steps=50)

```


### Citation

Please cite the following paper
```
To be added
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

This project is a port of the R version of [`ReMoDe`](https://github.com/hvdmaas/remode) . It is maintained by the [ODISSEI Social Data
Science (SoDa)](https://odissei-data.nl/nl/soda/) team.

<img src="soda_logo.png" alt="SoDa logo" width="250px"/>

Do you have questions, suggestions, or remarks? File an issue in the issue
tracker!