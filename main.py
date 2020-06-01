import numpy as np
from cec2013.cec2013 import *
from modules import *
import argparse


def main(func_id):
    f = CEC2013(func_id)
    d = f.get_dimension()
    pop_size = 16 * d
    archive = np.empty((0, d + 1))
    tot_eva = f.get_maxfes()
    # tot_eva *= 10
    v = 1
    for i in range(f.get_dimension()):
        v *= f.get_ubound(i) - f.get_lbound(i)
    pre_p = np.empty((0, d))
    radius = np.array([])
    while tot_eva > 0:
        P = reject_init(f, 4 * pop_size, pre_p, radius)
        tot_eva -= P.shape[0]
        score = np.zeros(P.shape[0])
        for i in range(P.shape[0]):
            score[i] = f.evaluate(P[i])
        pos = np.argsort(score)[::-1]
        P = P[pos[:1 * pop_size]]
        temp_p = np.empty((0, d))
        temp_radius = np.array([])
        for p in P:
            p2, tot_eva = cmsa2_reject(np.reshape(p, (-1, d)), tot_eva, f, np.ones(1) * f.evaluate(p),
                                       pre_p, radius)
            dist = np.linalg.norm(p2[0] - p, 2)
            temp_radius = np.hstack((temp_radius, dist))
            temp_p = np.vstack((temp_p, p2[0]))
            archive, tot_eva = union(archive, p2, f, tot_eva)
        pre_p = temp_p
        radius = temp_radius
        count, seeds = how_many_goptima(archive[:, :-1], CEC2013(func_id), 10 ** -5)
        if count == f.get_no_goptima():
            return archive
    return archive


def experiment():
    for exp in range(50):
        for func_id in range(1, 21):
            print('START', func_id, 'RUN', exp)
            # ans = main4(func_id)
            ans = main(func_id)
            score = np.zeros(ans.shape[0])
            f = CEC2013(func_id)
            for i in range(ans.shape[0]):
                score[i] = f.evaluate(ans[i, :-1])
            ans = np.hstack((ans, score.reshape(ans.shape[0], 1)))
            np.savetxt('./ans/' + 'problem%03d' % (func_id) + 'run%03d' % (exp+1) + '.dat', ans)


if __name__ == '__main__':
    experiment()
