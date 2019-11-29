# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from scripts import ranges as R

range_common = [
    [['seed'], [x * 100 for x in range(4)]],
]

range_lambd = [
    [['algorithm', 'lambd'], [0, 0.1, 0.5, 0.9, 1.]],
]

range_lambd = R.merge_ranges(range_common,  range_lambd)




