import numpy as np

MIN_REWARD = -10000.
def lints(reward_func, variant):


    n = 3  # n_arms
    d = 2  # n_dim


    # Initialization histories
    theta = {i: np.zeros(d) for i in range(0, n)}
    y = {i: np.zeros(d) for i in range(0, n)}
    V = {i: np.zeros((d, d)) for i in range(0, n)}
    B = {i: np.eye(d, d) for i in range(0, n)}
    X = {i: [] for i in range(0, n)}
    r = {i: [] for i in range(0, n)}


    Xs = []
    actions = []
    rs_ts = []
    tot_r = 0

    # Hyperparmeter
    sig = variant['sigma']
    delta = variant['delta']
    T = variant['T']

    v = sig * np.sqrt(9 * d * np.log(T / delta))
    # print(v)
    for t in range(0, T):
        # observe X
        x = np.random.randn(2)

        ##check condition
        flag = True
        for i in range(0, n):
            if np.any(np.isnan(theta[i])) or np.any(np.isnan(B[i])):
                flag = False

        # compute
        if not flag:
            action = np.random.choice(np.arange(0, n))
        else:
            max_r = -10000.
            for a in range(n):
                theta_hat = np.random.multivariate_normal(theta[a], v ** 2 * V[a])  # theta_a
                r_hat = x @ theta_hat
                if r_hat > max_r:
                    max_r = r_hat
                    action = a

        rew = reward_func(action, x) + np.random.normal(0, 0.1)

        Xs.append(x)  # from 0,1, ....
        actions.append(action)
        tot_r += rew
        rs_ts.append(tot_r)

        X[action].append(x.T)
        r[action].append(rew)

        if t > 10:
            W = []
            for k in range(t + 1):
                if actions[k] == action:
                    W.append(1)
            W = np.diag(W)

            # update
            X_arr = np.array(X[action])
            B[action] += X_arr.T @ W @ X_arr

            r_arr = np.array(r[action])
            y[action] += X_arr.T @ W @ r_arr

            B_inv = np.linalg.inv(B[action])
            theta[action] = B_inv @ y[action]
            V[action] = B_inv

