"""Independent post-gate state/scalar/direction saved-path accounting.

The new producer and helpers are never imported. Crossed cells are arithmetic
on recorded states and directions; no learner, graph, RNG, solver or future
policy is executed. Only common-post, both-actual-Go records are crossed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

TOPS = ("real", "rewired", "ann")
REPS = ("raw", "identity")
METHODS = ("shared", "isolated", "feedback_selective", "balanced", "gated_balanced")
CONTROLS = ("isolated", "balanced", "feedback_selective", "shared")
FACTORS = ("state", "scalar", "direction")
SIGNS = np.asarray([1., 1., -1., -1.])


def demand(condition, label):
    if not condition: raise ValueError(label)


def close(actual, expected, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    demand(actual.shape == expected.shape and actual.dtype.kind in "biuf" and np.isfinite(actual).all(), label + " finite axes")
    if expected.dtype.kind in "biu":
        demand(actual.dtype.kind in "biu" and np.array_equal(actual, expected), label + " exact")
    else: demand(np.allclose(actual, expected, atol=1e-9, rtol=1e-9), label + " arithmetic")


def compare_tree(actual, expected, label="summary"):
    if isinstance(expected, dict):
        demand(isinstance(actual, dict) and actual.keys() == expected.keys(), label + " keys")
        for key in expected: compare_tree(actual[key], expected[key], label + "." + key)
    elif isinstance(expected, list):
        demand(isinstance(actual, list) and len(actual) == len(expected), label + " length")
        for i, value in enumerate(expected): compare_tree(actual[i], value, label + "." + str(i))
    elif type(expected) is bool: demand(type(actual) is bool and actual == expected, label + " flag")
    elif type(expected) is int: demand(type(actual) is int and actual == expected, label + " integer")
    elif type(expected) is float:
        demand(type(actual) in (float, int) and np.isfinite(actual) and np.isclose(actual, expected, atol=1e-9, rtol=1e-9), label + " numeric")
    else: demand(actual == expected, label + " value")


def sigmoid(values):
    return 1. / (1. + np.exp(-np.clip(values, -40., 40.)))


def independent_bridge(z_control, z_gated, g_control, g_gated, f_control, f_gated):
    zc, zg, gc, gg, fc, fg = map(np.asarray, (z_control, z_gated, g_control, g_gated, f_control, f_gated))
    demand(zc.shape == zg.shape == fc.shape == fg.shape == gc.shape + (4,) and gc.shape == gg.shape
           and all(value.dtype.kind in "fiu" and np.isfinite(value).all() for value in (zc, zg, gc, gg, fc, fg)), "Recorded eight-corner finite axes")
    cells = np.empty(gc.shape + (8, 4))
    for code in range(8):
        z, g, f = (zg if code & 4 else zc), (gg if code & 2 else gc), (fg if code & 1 else fc)
        cells[..., code, :] = SIGNS * (sigmoid(z + g[..., None] * f) - sigmoid(z))
    return cells


def independent_shapley(cells):
    value = np.asarray(cells)
    demand(value.ndim >= 2 and value.shape[-2:] == (8, 4) and np.isfinite(value).all(), "Eight finite exact probability cells")
    output = []
    for bit in (4, 2, 1):
        other = [item for item in (4, 2, 1) if item != bit]
        allocation = (value[..., bit, :] - value[..., 0, :]) / 3.
        allocation += ((value[..., bit | other[0], :] - value[..., other[0], :])
                      + (value[..., bit | other[1], :] - value[..., other[1], :])) / 6.
        allocation += (value[..., 7, :] - value[..., 7 ^ bit, :]) / 3.
        output.append(allocation)
    result = np.stack(output, axis=-2)
    close(result.sum(axis=-2), value[..., 7, :] - value[..., 0, :], "Symmetric Shapley efficiency")
    return result
