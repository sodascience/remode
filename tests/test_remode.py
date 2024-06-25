"""Tests for the ReMoDe class."""
import numpy as np
from remode import ReMoDe, perform_fisher_test, perform_binomial_test


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


def custom_alpha_function():
    remode = ReMoDe(alpha=0.05, alpha_correction=lambda len, alpha: 0.1)
    assert remode._create_alpha_correction(10, 0.1) == 0.1


def test_remode_initialization():
    remode = ReMoDe(
        alpha=0.05, alpha_correction="none", statistical_test=perform_fisher_test
    )
    assert remode.alpha == 0.05
    assert remode._create_alpha_correction(10, 0.05) == 0.05
    assert remode.statistical_test == perform_fisher_test

    remode = ReMoDe(
        alpha=0.05, alpha_correction="max_modes", statistical_test=perform_fisher_test
    )
    assert remode._create_alpha_correction(10, 0.05) == 0.01
    assert remode.statistical_test == perform_fisher_test


def test_format_data():
    remode = ReMoDe()
    xt = [1, 2, 2, 3, 3, 3]
    formatted_data = remode.format_data(xt)
    assert np.array_equal(formatted_data, np.array([1, 2, 3]))


def test_fit():
    remode = ReMoDe()
    x = np.array([1, 2, 30, 2, 1])
    maxima = remode.fit(x)
    assert np.array_equal(maxima["modes"], np.array([2]))

    x = np.array([30, 2, 1, 2, 1])
    maxima = remode.fit(x)
    assert np.array_equal(maxima["modes"], np.array([0]))

    x = np.array([30, 2, 1, 2, 30])
    maxima = remode.fit(x)
    assert np.array_equal(maxima["modes"], np.array([0, 4]))


def test_jackknife():
    remode = ReMoDe()
    remode.xt = np.array([1, 2, 3, 2, 1])
    resampled_data = remode._jackknife(20)
    assert len(resampled_data) == 5


def test_evaluate_stability():
    remode = ReMoDe()
    remode.xt = np.array([1, 2, 20, 2, 1])
    remode.modes = np.array([2])
    result = remode.evaluate_stability(iterations=100, percentage_steps=5)
    print(result["stable_until"])
    assert "location_stability" in result
    assert "stable_until" in result
    assert result["stable_until"] == 75
