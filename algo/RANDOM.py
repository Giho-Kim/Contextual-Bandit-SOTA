import numpy as np

def random(reward_func, variant):
    n = 3  # n_arms
    d = 2  # n_dim

    T = variant['T']
    rs_random = []
    tot_r = 0
    for t in range(0, T):
        x = np.random.randn(d)

        action = np.random.choice(np.arange(0, n))

        rew = reward_func(action, x) + np.random.normal(0, 0.1)
        tot_r += rew

        rs_random.append(tot_r)
