#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: display_utils.py
Date: 2019/8/18 10:53 AM
"""
from IPython import display
import matplotlib.pyplot as plt


def use_svg_display():
    """ use svg to display plot"""
    display.set_matplotlib_formats("svg")


def set_figure_size(figsize=(3.5, 2.5)):
    """ set figure size"""
    use_svg_display()
    plt.rcParams["figure.figsize"] = figsize


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    """plot x and log(y)"""
    set_figure_size(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=":")
        plt.legend(legend)
    plt.show()
