# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
from scripts import ranges as R


range_common = [
    [['seed'], [x * 100 for x in range(16)]],
]

range_lambd = [
    [['algorithm', 'lambd'], [0., 0.1, 0.5, 0.9, 1.]],
    [['n_experts'], [1, 4, 8]]
]

range_lambd = R.merge_ranges(range_common, range_lambd)

range_pg = [
    [['algorithm', 'lambd'], [0.98]],
    [['use_experts'], [False]],
]

range_pg = R.merge_ranges(range_common, range_pg)


range_uniform = [
    [['algorithm', 'lambd'], [0., 0.1, 0.5, 0.9, 1.]],
    [['n_experts'], [1,4,8]],
    [['algorithm', 'uniform'], [True]],
    [['algorithm', 'policy_as_expert'], [False]],
]

range_uniform = R.merge_ranges(range_common, range_uniform)

range_aggrevate = [
    [['algorithm', 'lambd'], [0, 0.1, 0.5, 0.9, 1.]],
    [['n_experts'], [1]],
    [['algorithm', 'uniform'], [False]],
    [['algorithm', 'policy_as_expert'], [False]],
]

range_aggrevate = R.merge_ranges(range_common, range_aggrevate)




