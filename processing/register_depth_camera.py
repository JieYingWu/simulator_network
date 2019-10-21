# This scripts finds the transformation between the point cloud and the phantom
# Example: python register_depth_camera.py test_filtered/test2.ply ../../sofa/meshes/gel_phantom_1.ply 


import sys
import numpy as np

import pcl
#from pcl import IterativeClosestPoint

transform = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, -0.5984601, -0.8011526, 0.0],
                      [0.0, -0.8011526, -0.5984601, 0.0],
                      [0.0, 0.0,  0.0,  1.0]])

source = pcl.load(sys.argv[1])
source = pcl.PointCloud(source.to_array()*1000)
target = pcl.load(sys.argv[2])

#icp = source.make_IterativeClosestPoint()
#converged, transf, estimate, fitness = icp.icp(source, target, max_iter=2000)

model = pcl.SampleConsensusModelRegistration(source)
ransac = pcl.RandomSampleConsensus(model)
#ransac.set_threshold(.01)
#ransac.computeModel()

print(converged)
print(transf)
print(estimate)
print(fitness)
