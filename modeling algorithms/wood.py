from __future__ import division

import os
from scipy.spatial import distance
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

import microstructpy as msp


def generate_wood(path, id):
    # Colors
    c1 = '#12C2E9'
    c2 = '#C471ED'
    c3 = '#F64F59'

    # Offset
    off = 1

    # Create Directory
    dirname = os.path.join(os.path.dirname(__file__), 'fig')
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # Create Domain
    domain = msp.geometry.Rectangle(width=5, length=5)

    # Create Unpositioned Seeds
    phase2 = {'color': c1}
    ell_geom = msp.geometry.Circle(r=np.random.uniform(low=0.8, high=1.2))
    ell_geom2 = msp.geometry.Circle(r=np.random.uniform(low=0.8, high=1.2))

    # create center points
    lims = np.array(domain.limits)
    centers = np.zeros((2, 2))
    for i in range(2):
        f = np.random.rand(i + 1, 2)
        # f = np.random.uniform(low=0, high=1, size=[i + 1, 2])
        # f = np.random.randint(low=1, high=10, size=[i+1, 2]) * 0.1
        pts = f * lims[:, 0] + (1 - f) * lims[:, 1]
        try:
            min_dists = distance.cdist(pts, centers[:i]).min(axis=1)
            i_max = np.argmax(min_dists)
        except ValueError:  # this is the case when i=0
            i_max = 0
        centers[i] = pts[i_max]

    ell_seed = msp.seeding.Seed(ell_geom, phase=2, position=centers[0])
    ell_seed2 = msp.seeding.Seed(ell_geom2, phase=2, position=centers[1])

    mu = 0.9
    bnd = 0.15
    d_dist = scipy.stats.uniform(loc=mu-bnd, scale=2*bnd)

    phase0 = {'color': c2, 'shape': 'circle', 'd': d_dist}
    phase1 = {'color': c3, 'shape': 'circle', 'd': d_dist}
    circle_area = domain.area - ell_geom.area
    seeds = msp.seeding.SeedList.from_info([phase0, phase1], circle_area)

    seeds.append(ell_seed)
    seeds.append(ell_seed2)
    hold = [False for seed in seeds]
    hold[-1] =hold[-2] = True
    phases = [phase0, phase1, phase2]

    # Create Positioned Seeds
    seeds.position(domain, hold=hold, verbose=True)

    # Create Polygonal Mesh
    plt.figure(figsize=[1, 1], dpi=128)
    pmesh = msp.meshing.PolyMesh.from_seeds(seeds, domain)

    # Plot Polygonal Mesh
    pmesh.plot(edgecolors='k', facecolors="white", linewidth=3.5)


    # Set Up Axes

    xlim, ylim = domain.limits*0.95

    plt.xticks([])
    plt.yticks([])
    plt.box(False)
    plt.axis('off')
    plt.axis('square')
    plt.ylim(xlim)
    plt.xlim(ylim)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    plt.savefig(path + '\%d.png' % id)
    plt.close()


def main():
    path = r"fig"
    for i in range(100):
        generate_wood(path, id=i)


if __name__ == '__main__':
    main()