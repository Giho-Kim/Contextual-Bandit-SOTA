
from sklearn.linear_model import LogisticRegression
import numpy as np

MIN_REWARD = -10000.

def blts(reward_func, variant):

    n = 3  # n_arms
    d = 2  # n_dim


    # Hyperparameters
    T = variant['T']
    lam = variant['lam']  # regularization parameter
    thres = variant['thres']  # threshold for propensity score
    alpha = variant['alpha']  # reward distribution
    fill_buffer = variant['fill_buffer']

    # Initialization histories
    theta = {i: np.empty(d) for i in range(0, n)}
    V = {i: np.empty((d, d)) for i in range(0, n)}
    B = {i: np.empty((d, d)) for i in range(0, n)}
    X = {i: [] for i in range(0, n)}
    r = {i: [] for i in range(0, n)}

    for i in range(0, n):
        theta[i][:] = np.nan
        B[i][:] = np.nan

    Xs = []
    actions = []
    rs = []
    tot_r = 0


    for t in range(0, T):

        # observe context
        x = np.random.randn(d)

        # check condition
        flag = True
        for i in range(0, n):
            if np.any(np.isnan(theta[i])) or np.any(np.isnan(B[i])):
                flag = False

        # compute action
        if not flag:
            action = np.random.choice(np.arange(0, n))
        else:
            max_r = MIN_REWARD
            for a in range(n):
                theta_hat = np.random.multivariate_normal(theta[a], alpha ** 2 * V[a])  # theta_a
                r_hat = x @ theta_hat
                if r_hat > max_r:
                    max_r = r_hat
                    action = a

        # observe reward
        rew = reward_func(action, x) + np.random.normal(0, 0.1)

        # store
        Xs.append(x)  # from 0,1, ....
        actions.append(action)

        tot_r += rew
        rs.append(tot_r)

        X[action].append(x.T)
        r[action].append(rew)

        # obtain a Gaussian distribution and an upper confidence bound respectively
        # for the reward associated with each arm conditional on the context

        if t > fill_buffer:
            model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
            model.fit(Xs[:], actions[:])

            W = []
            for k in range(t + 1):
                if actions[k] == action:
                    feature = Xs[k]
                    prob = model.predict_proba([feature])[0]
                    prob = prob[action]
                    #                 print(thres, prob)
                    w = 1 / np.max([thres, prob])
                    W.append(w)
            W = np.diag(W)

            # update
            X_arr = np.array(X[action])
            B[action] = X_arr.T @ W @ X_arr + lam * np.eye(d)
            B_inv = np.linalg.inv(B[action])

            r_arr = np.array(r[action])
            theta[action] = B_inv @ X_arr.T @ W @ r_arr

            residual = (r_arr - X_arr @ theta[action]).reshape(-1, 1)
            V[action] = B_inv * (residual.T @ W @ residual)

