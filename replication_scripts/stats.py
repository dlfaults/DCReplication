#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import statsmodels.stats.power as pw


def cohen_d(orig_accuracy_list, accuracy_list):
    nx = len(orig_accuracy_list)
    ny = len(accuracy_list)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.std(orig_accuracy_list, ddof=1) ** 2 + (ny-1)*np.std(accuracy_list, ddof=1) ** 2) / dof)
    result = (np.mean(orig_accuracy_list) - np.mean(accuracy_list)) / pooled_std
    return result


def power(orig_accuracy_list, mutation_accuracy_list):
    eff_size = cohen_d(orig_accuracy_list, mutation_accuracy_list)
    pow = pw.FTestAnovaPower().solve_power(effect_size=eff_size, nobs=len(orig_accuracy_list) + len(mutation_accuracy_list), alpha=0.05)
    return pow
