import os
import sys
import pcl
import numpy as np

path = sys.argv[1]
files = os.listdir(path)

#### 2019-08-14-GelPhantom1 ####
#ransform = np.array([[0.529141, 0.3787912, -0.7592908, -34.48211],
#           [0.44376504, 0.6391739, 0.62811476, 18.01897],
#            [0.72324646, -0.6693094, 0.17012128, -19.62238],
#            [0., 0., 0., 1.]])

#### 2019-09-07-GelPhantom1 calibration.bag ####
# transform = np.array([[ 8.9053804e-01,  4.5363721e-01, -3.5208683e-02, -1.0271588e+01],
#                       [-5.6170724e-02,  1.8640044e-01,  9.8089284e-01, 1.3249355e+02],
#                       [ 4.5153180e-01, -8.7153816e-01,  1.9147959e-01, 2.3068979e+01],
#                       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]])


#### 2019-09-07-GelPhantom1 calibration2.bag ####
transform = np.array([[ 1.5453434e-01,  9.5433968e-01, -2.5568703e-01, -3.6765045e+01],
                      [-5.4349530e-02, -2.5019333e-01, -9.6667421e-01, -9.7587517e+01],
                      [-9.8650265e-01,  1.6327950e-01,  1.3204203e-02, -1.5522092e+01],
                      [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]])

for f in files:
    pc = pcl.load(path + f)

    # Segment the point cloud to extract phantom
    segment = pc.make_segmenter()
    segment.set_optimize_coefficients(True)
    segment.set_model_type(pcl.SACMODEL_PLANE)
    segment.set_method_type(pcl.SAC_RANSAC)
    segment.set_distance_threshold(0.025)
    indices,model = segment.segment()
    pc = pc.extract(indices, negative=True)

    segment = pc.make_segmenter()
    segment.set_optimize_coefficients(True)
    segment.set_model_type(pcl.SACMODEL_PLANE)
    segment.set_method_type(pcl.SAC_RANSAC)
    segment.set_distance_threshold(0.0025)
    indices,model = segment.segment()
    pc = pc.extract(indices, negative=False)
    
    # Transform the point cloud
    pc = pcl.PointCloud(pc.to_array()*1000)
    pc = np.dot(pc, np.transpose(transform[0:3, 0:3]))
    pc = pc + np.transpose(transform[0:3, 3])

    # Filter the point cloud
    pc = pcl.PointCloud(pc.astype(np.float32))
    # sor = pc.make_voxel_grid_filter()
    # sor.set_leaf_size(1, 1, 1)
    # pc_filtered = sor.filter()
    pc_filtered = pc
    
    pcl.save(pc_filtered, sys.argv[2] + f)
    
