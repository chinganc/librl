# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
from scripts.mamba import CONFIG
from scripts import configs as dc
from rl.core.utils.misc_utils import dict_update

config_dip = copy.deepcopy(CONFIG)
config_dip = dict_update(config_dip, dc.config_dip_traj)
config_dip['experimenter']['ro_kwargs']['max_n_rollouts']=8

config_humanoid = copy.deepcopy(CONFIG)
config_humanoid = dict_update(config_humanoid, dc.config_humanoid_traj)
config_humanoid['experimenter']['ro_kwargs']['max_n_rollouts']=50
config_humanoid['experimenter']['run_kwargs']['n_itrs']=3000


config_humanoid_expert = copy.deepcopy(CONFIG)
config_humanoid_expert = dict_update(config_humanoid_expert, dc.config_humanoid)
config_humanoid_expert['experimenter']['run_kwargs']['n_itrs']=3000
