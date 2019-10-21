import os
import sys
import pcl
import numpy as np

path = sys.argv[1]

pc = pcl.load(path)
segment = pc.make_segmenter()
segment.set_optimize_coefficients(True)
segment.set_model_type(pcl.SACMODEL_PLANE)
segment.set_method_type(pcl.SAC_RANSAC)
segment.set_distance_threshold(0.01)
indices,model = segment.segment()
pc = pc.extract(indices, negative=True)

segment = pc.make_segmenter()
segment.set_optimize_coefficients(True)
segment.set_model_type(pcl.SACMODEL_PLANE)
segment.set_method_type(pcl.SAC_RANSAC)
segment.set_distance_threshold(0.002)
indices,model = segment.segment()
pc = pc.extract(indices, negative=False)
pcl.save(pc,'test_filtered/test3.ply')
