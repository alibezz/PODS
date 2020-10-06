# PODS: Identification of Meaningful Outlier Relationships

This repository contains a simplified version of PODS, which is the method proposed  in paper
[Effective Discovery of Meaningful Outlier Relationships](https://arxiv.org/pdf/1910.08678.pdf).

## Requirements

* [Python 2.7](https://www.python.org/)
* [NumPy](https://numpy.org/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [SciPy](https://www.scipy.org/)
* [statsmodel](https://www.statsmodels.org/stable/index.html)
* [Matplotlib](https://matplotlib.org/) (for plots only)

## Execution

Given a folder `data` containing normalized temporal files (e.g., src/test_data); a `thresholds` file with values

a b c d

such that [a, b] is the positive interval within which every normalized value is considered an outlier, and [c, d]
is the negative interval within which every normalized value is considered an outlier (e.g., src/thresholds.txt); and values
for parameters alpha, beta, and rscore_min (e.g., 0.5, 0.67, and 0.25 respectively), PODS can be executed as follows:

$ cd src/
$ python find_good_correlations_from_normalized_data.py test_data/ thresholds.txt 0.5 0.67 0.25

and the output will state whether or not the outlier relationships across the temporal files are meaningful.