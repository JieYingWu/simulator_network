import sys
import numpy as np

import pcl
from pcl import IterativeClosestPoint

source = pcl.load(sys.argv[1])
target = pcl.load(sys.argv[2])

icp = source.make_IterativeClosestPoint()
converged, transf, estimate, fitness = icp.icp(source, target, max_iter=1000)

print(converged)
print(transf)
print(estimate)
print(fitness)
