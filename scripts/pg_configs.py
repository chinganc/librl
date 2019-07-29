import copy
from scripts.pg import CONFIG
from scripts import configs as dc
from rl.core.utils.misc_utils import dict_update

config_hopper = copy.deepcopy(CONFIG)
config_hopper = dict_update(config_hp, dc.config_hopper)
config_hopper['algorithm']['lr']=0.01
config_hopper['policy_units']=(64,64)

