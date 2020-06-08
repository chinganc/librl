# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Licensed under the MIT License.

from matplotlib import cm
from itertools import chain


SET1COLORS = cm.get_cmap('Set1').colors
SET2COLORS = cm.get_cmap('Set2').colors

COLOR = {
        'red': SET1COLORS[0],
        'blue': SET1COLORS[1],
        'lightblue': SET2COLORS[2],
        'green': SET1COLORS[2],
        'purple': SET1COLORS[3],
        'grey': SET1COLORS[8],
        'darkgreen': SET2COLORS[0],
        'orange': SET1COLORS[4],
        'pink': SET2COLORS[3],
        'lightgreen': SET2COLORS[4],
        'gold': SET2COLORS[5],
        'brown': SET1COLORS[6],
        }


mamba_configs = {
    'aggrevated': ('AggreVaTeD',  COLOR['green']),
    'mamba-0.9-max': ('MAMBA-0.9-max',  COLOR['red']),
    'mamba-0.9-mean': ('MAMBA-0.9-mean',  COLOR['blue']),
    'mamba-0.9-max(8)': ('MAMBA-0.9-max(8)',  COLOR['red']),
    'mamba-0.9-max(4)': ('MAMBA-0.9-max(4)',  COLOR['orange']),
    'mamba-0.9-max(2)': ('MAMBA-0.9-max(2)',  COLOR['lightblue']),
    'mamba-0.9-max(1)': ('MAMBA-0.9-max(1)',  COLOR['purple']),
    'mamba-0.9-max_': ('MAMBA-0.9-max',  COLOR['purple']),
    'mamba-0.5-max': ('MAMBA-0.5-max',  COLOR['gold']),
    'mamba-0.1-max': ('MAMBA-0.1-max',  COLOR['pink']),
    'pg-gae-0.9': ('PG-GAE-0.9', COLOR['grey']),
    'order': ['mamba-0.9-max', 'mamba-0.9-mean',
              'mamba-0.9-max(8)', 'mamba-0.9-max(4)', 'mamba-0.9-max(2)', 'mamba-0.9-max(1)',
              'mamba-0.9-max_', 'mamba-0.5-max', 'mamba-0.1-max',
              'aggrevated', 'pg-gae-0.9',
    ]
}


class Configs(object):
    def __init__(self, style=None, colormap=None):
        if not style:
            self.configs = None
            if colormap is None:
                c1 = iter(cm.get_cmap('Set1').colors)
                c2 = iter(cm.get_cmap('Set2').colors)
                c3 = iter(cm.get_cmap('Set3').colors)
                self.colors = chain(c1, c2, c3)
            else:
                self.colors = iter(cm.get_cmap(colormap).colors)
        else:
            self.configs = globals()[style + '_configs']
            for exp_name in self.configs['order']:
                assert exp_name in self.configs, 'Unknown exp: {}'.format(exp_name)

    def color(self, exp_name):
        if self.configs is None:
            color = next(self.colors)
        else:
            color = self.configs[exp_name][1]
        return color

    def label(self, exp_name):
        if self.configs is None:
            return exp_name
        return self.configs[exp_name][0]

    def sort_dirs(self, dirs):
        if self.configs is None:
            return dirs

        def custom_key(exp_name):
            if exp_name in self.configs['order']:
                return self.configs['order'].index(exp_name)
            else:
                return 100
        return sorted(dirs, key=custom_key)
