"""Tests for the ReMoDe class."""

import numpy as np
import pytest
from remode import (
    ReMoDe,
    perform_binomial_test,
    perform_bootstrap_test,
    perform_fisher_test,
)
from remode.remode import count_descriptive_peaks


def test_perform_fisher_test():
    x = np.array([10, 20, 30, 40, 50])
    candidate = 2
    left_min = 1
    right_min = 3
    p_left, p_right = perform_fisher_test(x, candidate, left_min, right_min)
    assert isinstance(p_left, float)
    assert isinstance(p_right, float)
    assert np.isclose(p_left, 0.08139159420716555)
    assert np.isclose(p_right, 0.9335662053784756)


def test_perform_binomial_test():
    x = np.array([10, 20, 30, 40, 50])
    candidate = 2
    left_min = 1
    right_min = 3
    p_left, p_right = perform_binomial_test(x, candidate, left_min, right_min)
    assert isinstance(p_left, float)
    assert isinstance(p_right, float)
    assert np.isclose(p_left, 0.10131937553227033)
    assert np.isclose(p_right, 0.9058918984884862)


def test_perform_bootstrap_test():
    x = np.array([10, 20, 30, 40, 50])
    candidate = 2
    left_min = 1
    right_min = 3

    p_left, p_right = perform_bootstrap_test(
        x, candidate, left_min, right_min, n_boot=2000, rng=np.random.default_rng(123)
    )
    assert isinstance(p_left, float)
    assert isinstance(p_right, float)
    assert np.isclose(p_left, p_right)
    assert 0 <= p_left <= 1


def custom_alpha_function():
    remode = ReMoDe(alpha=0.05, alpha_correction=lambda len, alpha: 0.1)
    assert remode._create_alpha_correction(10, 0.1) == 0.1


def test_remode_initialization():
    remode = ReMoDe(alpha=0.05, alpha_correction="none", statistical_test="fisher")
    assert remode.alpha == 0.05
    assert remode._create_alpha_correction(np.arange(10), 0.05) == 0.05
    assert remode.sign_test == "fisher"

    remode = ReMoDe(alpha=0.05, alpha_correction="max_modes", statistical_test="fisher")
    assert remode._create_alpha_correction(np.arange(10), 0.05) == 0.01
    assert remode.sign_test == "fisher"

    remode = ReMoDe(alpha=0.05, n_boot=1000, random_state=123)
    assert remode._create_alpha_correction(np.array([0, 2, 0, 3, 0]), 0.05) == 0.025
    assert remode.sign_test == "bootstrap"
    assert remode.definition == "shape_based"


def test_invalid_initialization_options():
    with pytest.raises(ValueError):
        ReMoDe(statistical_test="invalid")

    with pytest.raises(ValueError):
        ReMoDe(definition="invalid")

    with pytest.raises(ValueError):
        ReMoDe(n_boot=0)


def test_count_descriptive_peaks():
    assert count_descriptive_peaks(np.array([0, 2, 0, 3, 0])) == 2
    assert count_descriptive_peaks(np.array([1, 1, 1, 1])) == 0


def test_format_data():
    remode = ReMoDe()
    xt = [1, 2, 2, 3, 3, 3]
    formatted_data = remode.format_data(xt)
    assert np.array_equal(formatted_data, np.array([1, 2, 3]))


def test_fit():
    remode = ReMoDe(statistical_test="fisher")
    x = np.array([1, 2, 30, 2, 1])
    maxima = remode.fit(x)
    assert np.array_equal(maxima["modes"], np.array([2]))
    assert len(maxima["p_values"]) == 1
    assert len(maxima["approx_bayes_factors"]) == 1
    assert maxima["p_values"][0] < 0.05
    assert maxima["approx_bayes_factors"][0] > 1

    x = np.array([30, 2, 1, 2, 1])
    maxima = remode.fit(x)
    assert np.array_equal(maxima["modes"], np.array([0]))
    assert len(maxima["p_values"]) == 1
    assert len(maxima["approx_bayes_factors"]) == 1

    x = np.array([30, 2, 1, 2, 30])
    maxima = remode.fit(x)
    assert np.array_equal(maxima["modes"], np.array([0, 4]))
    assert len(maxima["p_values"]) == 2
    assert len(maxima["approx_bayes_factors"]) == 2


def test_fit_returns_r_parity_statistics_fisher():
    remode = ReMoDe(statistical_test="fisher")
    x = np.array([70, 80, 110, 30, 70, 100, 90, 120])

    result = remode.fit(x)

    assert np.array_equal(result["modes"], np.array([2, 7]))
    assert np.isclose(result["p_values"][0], 1.129097e-13, rtol=1e-5)
    assert np.isclose(result["approx_bayes_factors"][0], 1.0929e11, rtol=1e-5)
    assert result["p_values"][1] == 0
    assert np.isinf(result["approx_bayes_factors"][1])


def test_fit_returns_r_parity_statistics_binomial():
    remode = ReMoDe(statistical_test="binomial")
    x = np.array([70, 80, 110, 30, 70, 100, 90, 120])

    result = remode.fit(x)

    assert np.array_equal(result["modes"], np.array([2, 7]))
    assert np.isclose(result["p_values"][0], 3.12457e-12, rtol=1e-5)
    assert np.isclose(result["p_values"][1], 7.52316e-37, rtol=1e-5)
    assert np.isclose(result["approx_bayes_factors"][0], 4.44432e9, rtol=1e-5)
    assert np.isclose(result["approx_bayes_factors"][1], 5.87893e33, rtol=1e-5)


def test_jackknife():
    remode = ReMoDe(statistical_test="fisher")
    remode.xt = np.array([1, 2, 3, 2, 1])
    resampled_data = remode._jackknife(20)
    assert len(resampled_data) == 5


def test_peak_based_definition_uniform_distribution():
    x = np.array([10, 10, 10, 10, 10])
    shape_based = ReMoDe(statistical_test="fisher", definition="shape_based").fit(x)
    peak_based = ReMoDe(statistical_test="fisher", definition="peak_based").fit(x)

    assert shape_based["nr_of_modes"] > 0
    assert shape_based["uniform_distribution"] is None
    assert shape_based["uniformity_p_value"] is None
    assert peak_based["nr_of_modes"] == 0
    assert bool(peak_based["uniform_distribution"])
    assert peak_based["uniformity_p_value"] > 0.05


def test_peak_based_uniformity_uses_fixed_005_threshold():
    # p-value is ~0.13: above 0.05 but below 0.20
    x = np.array([1, 1, 1, 1, 5])
    result = ReMoDe(
        statistical_test="fisher",
        definition="peak_based",
        alpha=0.20,  # should not affect the uniformity cutoff
    ).fit(x)
    assert result["uniformity_p_value"] > 0.05
    assert result["uniformity_p_value"] < 0.20
    assert bool(result["uniform_distribution"])
    assert result["nr_of_modes"] == 0


def test_remode_stability():
    remode = ReMoDe(statistical_test="fisher")
    remode.xt = np.array([1, 2, 20, 2, 1])
    remode.modes = np.array([2])

    np.random.seed(123)
    result = remode.remode_stability(iterations=100, percentage_steps=5, plot=False)

    assert "location_stability" in result
    assert "stable_until" in result
    assert 0 <= result["stable_until"] <= 100

    np.random.seed(123)
    with pytest.warns(DeprecationWarning):
        alias_result = remode.evaluate_stability(
            iterations=100, percentage_steps=5, plot=False
        )

    assert alias_result["stable_until"] == result["stable_until"]
