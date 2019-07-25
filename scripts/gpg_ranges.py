import copy
from scripts import ranges as R


range_common = [
    [['seed'], [x * 100 for x in range(16)]],
]

range_lambd = [
    [['use_expert'], [True, False]],
    [['algorithm', 'lambd'], [0, 0.1, 0.5, 0.9, 1.]],
]

range_lambd = R.merge_ranges(range_common, range_lambd)




