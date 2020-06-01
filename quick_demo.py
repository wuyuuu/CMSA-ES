import numpy as np
from cec2013.cec2013 import *
from modules import *
import argparse


def main(func_id):
    f = CEC2013(func_id)
    d = f.get_dimension()
    pop_size = 16 * d
    archive = np.empty((0, d))
    tot_eva = f.get_maxfes()
    # tot_eva *= 10
    v = 1
    for i in range(f.get_dimension()):
        v *= f.get_ubound(i) - f.get_lbound(i)
    pre_p = np.empty((0, d))
    radius = np.array([])
    while tot_eva > 0:
        # P = simple_init(pop_size * 1, f)
        P = reject_init(f, 4 * pop_size, pre_p, radius)
        tot_eva -= P.shape[0]
        score = np.zeros(P.shape[0])
        for i in range(P.shape[0]):
            score[i] = f.evaluate(P[i])
        pos = np.argsort(score)[::-1]
        P = P[pos[:1 * pop_size]]
        # P = max_dist_select(P, pop_size)
        temp_p = np.empty((0, d))
        temp_radius = np.array([])
        for p in P:
            # p2, tot_eva = cmsa2_reject(np.reshape(p, (-1, d)), tot_eva, f, np.ones(1) * f.evaluate(p),
            #                            np.vstack((pre_p, temp_p)), np.hstack((radius, temp_radius)))
            p2, tot_eva = cmsa2_reject(np.reshape(p, (-1, d)), tot_eva, f, np.ones(1) * f.evaluate(p),
                                       pre_p, radius)
            dist = np.linalg.norm(p2[0] - p, 2)
            temp_radius = np.hstack((temp_radius, dist))
            temp_p = np.vstack((temp_p, p2[0]))
            # p1, tot_eva = cmsa(np.reshape(p, (-1, d)), tot_eva, f, np.ones(1) * f.evaluate(p))
            archive, tot_eva = union(archive, p2, f, tot_eva)
            # archive = union(archive, p1, f)
        print('archive shape,', archive.shape)
        print('eva remain,', tot_eva)
        see_result(archive[:, :-1], f)
        pre_p = temp_p
        radius = temp_radius
    return archive


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--func_id',default=1,type=int)
    args = parse.parse_args()
    func_id = args.func_id
    ans = main(func_id)
    see_result(ans[:, :-1], CEC2013(func_id))
    np.savetxt('points.txt', ans)
