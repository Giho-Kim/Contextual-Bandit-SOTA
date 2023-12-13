from configs.default import default_config
from algo import *

import click
import os
import json

def reward_func(arm, x):
    if arm == 0:
        return 0.5 * (x[0] + 1) ** 2 + 0.5 * (x[1] + 1) ** 2
    elif arm == 1:
        return 1
    elif arm == 2:
        return 2 - 0.5 * (x[0] + 1) ** 2 - 0.5 * (x[1] + 1) ** 2


def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

def experiment(variant):
    algo = variant['algo']
    if algo == 'LinTS':
        lints(reward_func, variant)
    elif algo == 'random':
        random(reward_func, variant)
    elif algo == 'BLTS':
        blts(reward_func, variant)
    else :
        print("not existed")

@click.command()
@click.argument('config', default=None)
def main(config):
    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    experiment(variant)




if __name__ == '__main__':
    main()
