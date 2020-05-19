# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
from scripts import ranges as R


range_common = [
    [['seed'], [x * 100 for x in range(8)]],
]


# basic baseline

range_pg = [
    [['top_log_dir'], ['log_pg']],
    [['algorithm', 'lambd'], [0.98, 1.00]],
    [['use_experts'], [False]],
]
range_pg = R.merge_ranges(range_common, range_pg)

range_aggrevated = [
    [['top_log_dir'], ['log_aggrevated']],
    [['algorithm', 'lambd'], [0, 0.1, 0.5, 0.9, 1.]],
    [['n_experts'], [1]],
    [['algorithm', 'strategy'], ['mean']],
    [['algorithm', 'policy_as_expert'], [False]],
]
range_aggrevated = R.merge_ranges(range_common, range_aggrevated)


# aggregation

range_lambd = [
    [['top_log_dir'], ['log_mamba']],
    [['algorithm', 'lambd'], [0., 0.1, 0.5, 0.9, 0.98]],
    [['n_experts'], [1, 2, 4, 8]],
    [['algorithm', 'strategy'], ['max']],
    [['algorithm', 'policy_as_expert'], [False]],
]
range_lambd = R.merge_ranges(range_common, range_lambd)

range_uniform = [
    [['top_log_dir'], ['log_uniform_s']],
    [['algorithm', 'lambd'], [0., 0.1, 0.5, 0.9, 0.98]],
    [['n_experts'], [1, 2, 4, 8]],
    [['algorithm', 'strategy'], ['uniform']],
    [['algorithm', 'policy_as_expert'], [True]],
]
range_uniform = R.merge_ranges(range_common, range_uniform)

range_mean = [
    [['top_log_dir'], ['log_mean_s']],
    [['algorithm', 'lambd'], [0., 0.1, 0.5, 0.9, 0.98]],
    [['n_experts'], [1, 2, 4, 8]],
    [['algorithm', 'strategy'], ['mean']],
    [['algorithm', 'policy_as_expert'], [True]],
]
range_mean = R.merge_ranges(range_common, range_mean)

# debug

range_debug = [
    [['top_log_dir'], ['log_debug']],
    [['seed'], [x * 100 for x in range(3)]],
    [['algorithm', 'lambd'], [0., 0.5]],
    [['algorithm', 'policy_as_expert'], [False]],
    [['algorithm', 'strategy'], ['max', 'mean', 'uniform']],
    [['n_experts'], [1]]
]





