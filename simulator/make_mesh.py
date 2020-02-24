import sys
import numpy as np

dim = np.array([68.7, 35.8, 39.3])
#nodes = np.array([23,12,13])
nodes = np.array([13,5,5])
step = dim/(nodes-1)

vol = np.zeros((nodes[0]*nodes[1]*nodes[2], 3))

pt = 0
for i in range(nodes[0]):
    x = -dim[0]/2 + step[0]*i
    for j in range(nodes[1]):
        y = -dim[1]/2 + step[1]*j
        for k in range(nodes[2]):
            z = -dim[2]/2 + step[2]*k
            vol[pt] = [x,y,z]
            pt += 1

if len(sys.argv) > 1:
    for i in range(int(sys.argv[2])):
        np.savetxt(sys.argv[1] + "/position" + '%04d' % (i) + ".txt", vol)
else:
    np.savetxt("mesh.txt", vol)
