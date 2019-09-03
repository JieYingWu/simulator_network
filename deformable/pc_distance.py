import pymesh

mesh = pymesh.load_mesh('lego_support.obj')
pc = pymesh.load_mesh('data0/1565794994259.ply').vertices
squared_distances, face_indices, closest_points = pymesh.distance_to_mesh(mesh, pc)
print(squared_distances[0:20])
