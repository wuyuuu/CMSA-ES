import numpy as np
from cec2013.cec2013 import *
import collections


def hv_test(x1, x2, f, N=5):
    v = min(f.evaluate(x1), f.evaluate(x2))

    data = np.linspace(x1, x2, N + 2)
    for i in range(1, N + 1):
        if f.evaluate(data[i]) < v:
            return False, i

    return True, N


def nearest_hv_test(x, P, f):
    if P.shape[0] == 0:
        return False, 0
    dist = np.linalg.norm(P - x, axis=1)
    idx = np.argmin(dist)
    return hv_test(x, P[idx], f)


def union(archive, p, f, tot_eva):
    eva_cost = 0
    if p.shape[0] == 0:
        return archive
    score = np.zeros(p.shape[0])
    for i in range(p.shape[0]):
        score[i] = f.evaluate(p[i])
    pos = np.argsort(score)[::-1]
    p = p[pos]

    if archive.shape[0] == 0:
        temp = np.empty((0, p.shape[1]))
        for i in range(p.shape[0]):
            flag, eva = nearest_hv_test(p[i], temp, f)
            eva_cost += eva
            if not flag:
                temp = np.vstack((temp, p[i]))
        return np.hstack((temp, np.reshape(np.ones(temp.shape[0]) * tot_eva, (-1, 1)))), tot_eva - eva_cost
    max_score = np.max(score)
    for i in range(archive.shape[0]):
        max_score = max(f.evaluate(archive[i, :-1]), max_score)
    ans = np.empty((0, f.get_dimension() + 1))
    for i in range(archive.shape[0]):
        if f.evaluate(archive[i, :-1]) < max_score - 1e-1:
            continue
        ans = np.vstack((ans, archive[i]))
    archive = ans
    for i in range(p.shape[0]):
        if score[pos[i]] < max_score - 1e-1:
            continue
        flag, eva = nearest_hv_test(p[i], archive[:, :-1], f)
        eva_cost += eva
        if not flag:
            archive = np.vstack((archive, np.hstack((p[i], tot_eva))))
    return archive, tot_eva - eva_cost


def max_dist_select(x, pop_size):
    n, d = x.shape[0], x.shape[1]
    dist = np.zeros(n)
    for i in range(n):
        dist[i] = np.max(np.linalg.norm(np.delete(x, i, axis=0) - x[i], 2, axis=1))
    pos = np.argsort(dist)[::-1]
    return x[pos[:pop_size]]


def scatter_init(n, pop_size, f):
    d = f.get_dimension()
    x = np.zeros((n, d))
    ub = np.zeros(d)
    lb = np.zeros(d)
    for k in range(d):
        ub[k] = f.get_ubound(k)
        lb[k] = f.get_lbound(k)
    for i in range(n):
        x[i] = lb + (ub - lb) * np.random.rand(1, d)
    score = np.zeros(n)
    for i in range(n):
        score[i] = f.evaluate(x[i])
    pos = np.argsort(score)[::-1]
    return x[pos[:pop_size]]
    return max_dist_select(x[pos[:pop_size]], pop_size // 2)


def reject_init(f, pop_size, archive, radius):
    dim = f.get_dimension()
    X = np.zeros((pop_size, dim))
    ub = np.zeros(dim)
    lb = np.zeros(dim)
    # Get lower, upper bounds
    for k in range(dim):
        ub[k] = f.get_ubound(k)
        lb[k] = f.get_lbound(k)
    # Create population within bounds
    for j in range(pop_size):
        X[j] = lb + (ub - lb) * np.random.rand(1, dim)
        flag = True
        for k in range(archive.shape[0]):
            if np.linalg.norm(archive[k] - X[j], 2) < radius[k]:
                flag = False
                break
        while flag == False and np.random.rand() < 0.9:
            # print(flag)
            X[j] = lb + (ub - lb) * np.random.rand(1, dim)
            flag = True
            for k in range(archive.shape[0]):
                if np.linalg.norm(archive[k] - X[j], 2) < radius[k]:
                    flag = False
                    break
    return X
    # return max_dist_select(X, pop_size // 2)


def simple_init(pop_size, f):
    dim = f.get_dimension()
    X = np.zeros((pop_size, dim))
    ub = np.zeros(dim)
    lb = np.zeros(dim)
    for k in range(dim):
        ub[k] = f.get_ubound(k)
        lb[k] = f.get_lbound(k)
    for j in range(pop_size):
        X[j] = lb + (ub - lb) * np.random.rand(1, dim)
    return X


def hv_test(x1, x2, f, N=5):
    # WARNING: Potential risk in "Line 13"
    if (np.linalg.norm(x1 - x2, 2) < 1e-7):
        return True, 0
    v = min(f.evaluate(x1), f.evaluate(x2))

    data = np.linspace(x1, x2, N + 2)
    # print(data)
    for i in range(1, N + 1):
        if f.evaluate(data[i]) < v:
            return False, i

    return True, N


def nearest_better_tree(P):
    # P has been sorted by its fitness
    has = collections.defaultdict(list)
    n, d = P.shape
    for i in range(n):
        temp = []
        for j in range(i):
            temp.append((j, np.linalg.norm(P[i] - P[j])))
        temp.sort(key=lambda x: x[1])
        if len(temp) < d + 1:
            has[i] = temp
        else:
            has[i] = temp[:d + 1]
    return has


def clustering(P, f):
    # P need be sorted
    d = f.get_dimension()
    V = 1
    for i in range(d):
        V *= np.abs(f.get_lbound(i) - f.get_ubound(i))
    # P has been sorted by its fitness
    n, d = P.shape
    C = []
    getC = {}
    has = nearest_better_tree(P)
    evaluation = 0

    for i in range(n // 2):
        flag = False
        temp = has[i]
        for j in range(min(i, d + 1)):
            pos, delta = temp[j]
            if pos not in getC:
                continue
            pos = C[getC[pos]][0]
            delta = np.linalg.norm(P[i] - P[pos])
            Nt = 2 + int(delta / ((V / n) ** (1 / d)))
            test, eva = hv_test(P[i], P[pos], f, Nt)
            evaluation += eva
            if test:
                flag = True
                C[getC[pos]].append(i)
                break
        if not flag:
            getC[i] = len(C)
            C.append([i])
    return C, evaluation


def Sample(mean, cov, lam, f, sigma):
    ans = np.array([])
    tau = 1 / np.sqrt(2 * f.get_dimension())
    d = f.get_dimension()
    next_sigma = np.zeros(lam)
    ub = np.zeros(d)
    lb = np.zeros(d)
    for i in range(d):
        ub[i] = f.get_ubound(i)
        lb[i] = f.get_lbound(i)
    for i in range(lam):
        x = np.random.multivariate_normal(np.zeros(d), cov)
        step = np.exp(tau * np.random.normal())
        next_sigma[i] = step * sigma
        x = mean + x * next_sigma[i]
        for j in range(d):
            if x[j] > ub[j]:
                x[j] = ub[j]
            if x[j] < lb[j]:
                x[j] = lb[j]
        if ans.shape[0] == 0:
            ans = np.reshape(x, (-1, d))
        else:
            ans = np.vstack((ans, x))
    return ans, next_sigma


def see_result(Archive, f):
    score = np.zeros(Archive.shape[0])
    for i in range(Archive.shape[0]):
        score[i] = f.evaluate(Archive[i])
    print('infor', f.get_info())
    print('best so far', np.max(score))
    for err in range(-1, -6, -1):
        count, seeds = how_many_goptima(Archive, f, 10 ** err)
        print("Let err be ", 10 ** err, " In the current population there exist", count, "global optimizers.")


def closest_hv_test(p, archive, f, N):
    if archive.shape[0] == 0:
        return False, 0
    dist = np.linalg.norm(archive - p, axis=1)
    pos = np.argmin(dist)
    return hv_test(p, archive[pos], f, N)


def reject_cmsa(p, tot_eva, f, now_score, elite_archive):
    d = f.get_dimension()
    mean = np.mean(p, axis=0)
    sigma = np.ones(p.shape[0])
    cov = np.eye(d)
    mu = int((4.5 + 3 * np.log(d)))
    MAX_DEPTH = 10 + int(30 * d / mu)
    depth = 0
    best_score = f.evaluate(p[0])
    elite_num = max(1, int(0.15 * mu))
    # print('-' * 30)

    while depth < MAX_DEPTH and tot_eva > 0 and np.linalg.cond(cov) < 10 ** 14:
        if depth > 0 and depth % 5 == 0:
            # if depth == 5:
            flag, eva = closest_hv_test(p[0], elite_archive, f, min(5, tot_eva))
            tot_eva -= eva
            if flag and np.random.rand():
                return p, tot_eva
        # print('sigma',np.mean(sigma))
        # print('cond',np.linalg.cond(cov))
        lam = min(4 * mu, tot_eva)
        pre_sigma_mean = np.mean(sigma)

        next_p, next_sigma = Sample(mean, cov, lam, f, np.mean(sigma))
        tot_eva -= lam

        next_score = np.zeros(next_p.shape[0])
        for i in range(next_p.shape[0]):
            next_score[i] = f.evaluate(next_p[i])

        p = np.vstack((p[:elite_num], next_p))
        sigma = np.hstack((sigma[:elite_num], next_sigma))
        now_score = np.hstack((now_score[:elite_num], next_score))

        pos = np.argsort(now_score)[::-1]
        if tot_eva == 0:
            p = p[pos]
            break

        p = p[pos]
        sigma = sigma[pos]
        now_score = now_score[pos[:mu]]

        if now_score[0] > best_score + 1e-6:
            depth = 0
            best_score = now_score[0]
        else:
            depth += 1
        x_mean = np.mean(p[:mu], axis=0)
        z = [(p[i] - x_mean) / sigma[i] for i in range(mu)]

        tau_c = 1 + d * (d + 1) / (2 * mu)
        cov *= (1 - 1 / tau_c)
        w = np.zeros(mu)
        for i in range(mu):
            w[i] = np.log(mu + 1) - np.log(i + 1)
        w = w / np.sum(w)
        for i in range(mu):
            cov = cov + 1 / tau_c * (w[i] * np.matmul(np.reshape(z[i], (-1, 1)), np.reshape(z[i], (1, -1))))
        mean = np.zeros_like(p[0])
        for i in range(mu):
            mean = mean + w[i] * p[i]
        tot = 1
        sub = 1
        for i in range(sigma.shape[0]):
            tot *= sigma[i] ** (1.0 / sigma.shape[0])
        for i in range(mu):
            sub *= sigma[i] ** (w[i])
        sigma = np.ones(mu) * pre_sigma_mean * sub / tot
    return p, tot_eva


def reject(p, archive, radius):
    if archive.shape[0] == 0:
        return False
    # print(archive.shape,len(radius))
    dist = np.linalg.norm(archive - p, 2, axis=1)
    idx = np.argmin(dist)
    return dist[idx] < radius[idx]


def cmsa2_reject(p, tot_eva, f, now_score, archive, radius):
    d = f.get_dimension()
    mean = np.mean(p, axis=0)
    sigma = np.ones(p.shape[0])
    cov = np.eye(d)
    # for i in range(d):
    #     cov[i, i] = (f.get_ubound(i) - f.get_lbound(i)) * 0.01
    mu = int((4.5 + 3 * np.log(d)))
    MAX_DEPTH = 10 + int(30 * d / mu)
    depth = 0
    best_score = f.evaluate(p[0])
    elite_num = max(1, int(0.15 * mu))
    # print('-' * 30)

    while depth < MAX_DEPTH and tot_eva > 0:
        if (depth != 0 and depth % 5 == 0) and reject(p[0], archive, radius):
            return p, tot_eva

        # print('sigma',np.mean(sigma))
        # print('cond',np.linalg.cond(cov))
        lam = min(4 * mu, tot_eva)
        pre_sigma_mean = np.mean(sigma)

        next_p, next_sigma = Sample(mean, cov, lam, f, np.mean(sigma))
        tot_eva -= lam

        next_score = np.zeros(next_p.shape[0])
        for i in range(next_p.shape[0]):
            next_score[i] = f.evaluate(next_p[i])

        p = np.vstack((p[:elite_num], next_p))
        sigma = np.hstack((sigma[:elite_num], next_sigma))
        now_score = np.hstack((now_score[:elite_num], next_score))

        pos = np.argsort(now_score)[::-1]
        if tot_eva == 0:
            p = p[pos[:elite_num]]
            break

        p = p[pos]
        sigma = sigma[pos]
        now_score = now_score[pos[:mu]]

        if now_score[0] > best_score + 1e-6:
            depth = 0
            best_score = now_score[0]
        else:
            depth += 1
        x_mean = np.mean(p[:mu], axis=0)
        # z = [(p[i] - x_mean) / sigma[i] for i in range(mu)]
        z = [(p[i] - x_mean) / sigma[i] for i in range(p.shape[0])]
        tau_c = 1 + d * (d + 1) / (2 * mu)

        w = np.zeros(mu)
        for i in range(mu):
            w[i] = np.log(mu + 1) - np.log(i + 1)
        w = w / np.sum(w)

        cov *= (1 - 1 / tau_c)
        for i in range(mu):
            cov = cov + 1 / tau_c * (w[i] * np.matmul(np.reshape(z[i], (-1, 1)), np.reshape(z[i], (1, -1))))
        # ss = p.shape[0]
        # for i in range(ss):
        #     cov = cov - (1 / (ss * tau_c)) * np.matmul(np.reshape(z[i], (-1, 1)), np.reshape(z[i], (1, -1)))
        if np.min(np.linalg.eigh(cov)[0]) < 0:
            return p, tot_eva

        mean = np.zeros_like(p[0])
        for i in range(mu):
            mean = mean + w[i] * p[i]
        tot = 1
        sub = 1
        for i in range(sigma.shape[0]):
            tot *= sigma[i] ** (1.0 / sigma.shape[0])
        for i in range(mu):
            sub *= sigma[i] ** (w[i])
        sigma = np.ones(mu) * pre_sigma_mean * sub / tot
    return p, tot_eva


def cmsa2(p, tot_eva, f, now_score):
    d = f.get_dimension()
    mean = np.mean(p, axis=0)
    sigma = np.ones(p.shape[0])
    cov = np.eye(d)
    # for i in range(d):
    #     cov[i, i] = (f.get_ubound(i) - f.get_lbound(i)) * 0.01
    mu = int((4.5 + 3 * np.log(d)))
    MAX_DEPTH = 10 + int(30 * d / mu)
    depth = 0
    best_score = f.evaluate(p[0])
    elite_num = max(1, int(0.15 * mu))
    # print('-' * 30)

    while depth < MAX_DEPTH and tot_eva > 0:

        # print('sigma',np.mean(sigma))
        # print('cond',np.linalg.cond(cov))
        lam = min(4 * mu, tot_eva)
        pre_sigma_mean = np.mean(sigma)

        next_p, next_sigma = Sample(mean, cov, lam, f, np.mean(sigma))
        tot_eva -= lam

        next_score = np.zeros(next_p.shape[0])
        for i in range(next_p.shape[0]):
            next_score[i] = f.evaluate(next_p[i])

        p = np.vstack((p[:elite_num], next_p))
        sigma = np.hstack((sigma[:elite_num], next_sigma))
        now_score = np.hstack((now_score[:elite_num], next_score))

        pos = np.argsort(now_score)[::-1]
        if tot_eva == 0:
            p = p[pos[:elite_num]]
            break

        p = p[pos]
        sigma = sigma[pos]
        now_score = now_score[pos[:mu]]

        if now_score[0] > best_score + 1e-6:
            depth = 0
            best_score = now_score[0]
        else:
            depth += 1
        x_mean = np.mean(p[:mu], axis=0)
        # z = [(p[i] - x_mean) / sigma[i] for i in range(mu)]
        z = [(p[i] - x_mean) / sigma[i] for i in range(p.shape[0])]
        tau_c = 1 + d * (d + 1) / (2 * mu)

        w = np.zeros(mu)
        for i in range(mu):
            w[i] = np.log(mu + 1) - np.log(i + 1)
        w = w / np.sum(w)

        cov *= (1 - 1 / tau_c)
        for i in range(mu):
            cov = cov + 1 / tau_c * (w[i] * np.matmul(np.reshape(z[i], (-1, 1)), np.reshape(z[i], (1, -1))))
        # ss = p.shape[0]
        # for i in range(ss):
        #     cov = cov - (1 / (ss * tau_c)) * np.matmul(np.reshape(z[i], (-1, 1)), np.reshape(z[i], (1, -1)))
        if np.min(np.linalg.eigh(cov)[0]) < 0:
            return p, tot_eva

        mean = np.zeros_like(p[0])
        for i in range(mu):
            mean = mean + w[i] * p[i]
        tot = 1
        sub = 1
        for i in range(sigma.shape[0]):
            tot *= sigma[i] ** (1.0 / sigma.shape[0])
        for i in range(mu):
            sub *= sigma[i] ** (w[i])
        sigma = np.ones(mu) * pre_sigma_mean * sub / tot
    return p, tot_eva


def cmsa(p, tot_eva, f, now_score):
    d = f.get_dimension()
    mean = np.mean(p, axis=0)
    sigma = np.ones(p.shape[0])
    cov = np.eye(d)

    mu = int((4.5 + 3 * np.log(d)))
    MAX_DEPTH = 10 + int(30 * d / mu)
    depth = 0
    best_score = f.evaluate(p[0])
    elite_num = max(1, int(0.15 * mu))
    # print('-' * 30)

    while depth < MAX_DEPTH and tot_eva > 0:
        # if np.linalg.cond(cov) > 10 ** 14:
        #     return p,tot_eva
        # print('sigma',np.mean(sigma))
        # print('cond',np.linalg.cond(cov))
        lam = min(4 * mu, tot_eva)
        pre_sigma_mean = np.mean(sigma)

        next_p, next_sigma = Sample(mean, cov, lam, f, np.mean(sigma))
        tot_eva -= lam

        next_score = np.zeros(next_p.shape[0])
        for i in range(next_p.shape[0]):
            next_score[i] = f.evaluate(next_p[i])

        p = np.vstack((p[:elite_num], next_p))
        sigma = np.hstack((sigma[:elite_num], next_sigma))
        now_score = np.hstack((now_score[:elite_num], next_score))

        pos = np.argsort(now_score)[::-1]
        if tot_eva == 0:
            p = p[pos]
            break

        p = p[pos]
        sigma = sigma[pos]
        now_score = now_score[pos[:mu]]

        if now_score[0] > best_score + 1e-6:
            depth = 0
            best_score = now_score[0]
        else:
            depth += 1
        x_mean = np.mean(p[:mu], axis=0)
        # z = [(p[i] - x_mean) / sigma[i] for i in range(mu)]
        z = [(p[i] - x_mean) / sigma[i] for i in range(p.shape[0])]
        tau_c = 1 + d * (d + 1) / (2 * mu)

        w = np.zeros(mu)
        for i in range(mu):
            w[i] = np.log(mu + 1) - np.log(i + 1)
        w = w / np.sum(w)

        # cov *= (1 - 1 / tau_c)
        for i in range(mu):
            cov = cov + 1 / tau_c * (w[i] * np.matmul(np.reshape(z[i], (-1, 1)), np.reshape(z[i], (1, -1))))
        ss = p.shape[0]
        for i in range(ss):
            cov = cov - (1 / (ss * tau_c)) * np.matmul(np.reshape(z[i], (-1, 1)), np.reshape(z[i], (1, -1)))
        if np.min(np.linalg.eigvalsh(cov)) < 0:
            # cov = np.eye(d)*np.max(np.linalg.eigh(cov)[0])
            # print(np.linalg.eigvalsh(cov))
            # cov = np.eye(d) * np.linalg.eigvalsh(cov)
            # cov = np.eye(d)
            # print(np.linalg.eigvalsh(cov))
            # print(np.linalg.cond(cov))
            return p, tot_eva
        mean = np.zeros_like(p[0])
        for i in range(mu):
            mean = mean + w[i] * p[i]
        tot = 1
        sub = 1
        for i in range(sigma.shape[0]):
            tot *= sigma[i] ** (1.0 / sigma.shape[0])
        for i in range(mu):
            sub *= sigma[i] ** (w[i])
        sigma = np.ones(mu) * pre_sigma_mean * sub / tot
    return p, tot_eva
