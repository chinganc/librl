import copy
from scripts import ranges as R


range_common = [
    [['seed'], [x * 100 for x in range(16)]],
]

range_lambd = [
    [['algorithm', 'lambd'], [0, 0.1, 0.5, 0.9, 1.]],
    [['use_experts'], [True, False]],
]

range_lambd = R.merge_ranges(range_common, range_lambd)

range_pg = [
    [['algorithm', 'lambd'], [0.98]],
    [['use_experts'], [False]],
]

range_pg = R.merge_ranges(range_common, range_pg)






