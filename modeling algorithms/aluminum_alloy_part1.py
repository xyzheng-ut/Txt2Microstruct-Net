import matplotlib.pyplot as plt
import microstructpy as msp
import scipy.stats
import numpy as np
from scipy.spatial import distance
import os


def set_box_array(x):
    y = []
    for i in range(5):  # z: layer1, 2, 3
        temp = np.zeros((len(x), 3))
        temp[:, 2] = 1*i
        temp_z = x + temp
        for j in range(5):  # x
            temp = np.zeros((len(x), 3))
            temp[:, 0] = 1 * j
            temp_x = temp_z + temp
            for k in range(5):  # y
                temp = np.zeros((len(x), 3))
                temp[:, 1] = 1 * k
                temp_y = temp_x + temp
                # print(temp_y)

                y.append(temp_y)

    y = np.reshape(np.array(y), [-1,3])
    return y


def save_region_facet_point(region, facet, point, file):
    with open(file, "a") as f:
        f.write("#regions: point index\n")
        for i in range(len(region)):
            f.write("region_%d," % (i + 1))
            np.savetxt(f, region[i], fmt="%.d", delimiter=",", newline=",")
            f.write("\n")
        f.write("#regions: point index\n")
        for i in range(len(facet)):
            f.write("facet_%d," % (i + 1))
            np.savetxt(f, facet[i], fmt="%.d", delimiter=",", newline=",")
            f.write("\n")
        f.write("#points_index, x, y, z\n")
        for i in range(len(point)):
            f.write("point_%d," % (i + 1))
            np.savetxt(f, point[i], fmt="%.6e", delimiter=",", newline=",")
            f.write("\n")


# Create domain
domain = msp.geometry.Box(corner=(0, 0, 0))

# Create list of seed points
factory = msp.seeding.Seed.factory
n = 27
numbers = 100

# Position seeds according to Mitchell's Best Candidate Algorithm

lims = np.array(domain.limits)  # [[0. 1.], [0. 1.], [0. 1.]]
np.random.seed(999)

for epoch in range(numbers):
    centers = np.zeros((n, 3))

    seeds = msp.seeding.SeedList([factory('sphere', r=0.1) for i in range(n*125)])
    for i in range(n):
        f = np.random.rand(i + 1, 3)
        pts = f #* lims[:, 0] + (1 - f) * lims[:, 1]
        try:
            min_dists = distance.cdist(pts, set_box_array(centers[:i])).min(axis=1)
            i_max = np.argmax(min_dists)
        except ValueError:  # this is the case when i=0
            i_max = 0
        centers[i] = pts[i_max]


    centers = set_box_array(centers)
    for i in range(n*125):
        seeds[i].position = (centers[i])/5

    # set pmesh, 3x3x3 box array
    pmesh = msp.meshing.PolyMesh.from_seeds(seeds, domain, edge_opt=True, n_iter=20)
    p_points = np.array(pmesh.points) * 5 #- 2

    pregion_reshape = np.reshape(np.array(pmesh.regions), [-1,n])
    pregion_center = pregion_reshape[62][:]


    new_point_list = []
    new_facet_list = []
    new_region_list = []
    for region in pregion_center:
        region_new = []
        for facet_index in region:
            facet = pmesh.facets[facet_index]
            facet_new = []
            for point_index in facet:
                if point_index not in new_point_list:
                    new_point_list.append(point_index)
                facet_new.append(new_point_list.index(point_index))
            if facet_new not in new_facet_list:
                new_facet_list.append(facet_new)
            region_new.append(new_facet_list.index(facet_new))
        new_region_list.append(region_new)


    point_ultimate = []
    for point in new_point_list:
        point_ultimate.append(p_points[point].tolist())


    # os.remove(r"face\face_%d.txt" % epoch)
    save_region_facet_point(new_region_list, new_facet_list, point_ultimate, r"face\%d.txt" % epoch)
