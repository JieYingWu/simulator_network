# Currently ignoring that quaternions shouldn't be interpolated linearly since we don't use it

import os
import sys
import numpy as np


kinematics_path = sys.argv[1]
kinematics_array = np.genfromtxt(kinematics_path, delimiter=',')
label_path = sys.argv[2]
label_files = os.listdir(label_path)
#label_array = np.genfromtxt(sys.argv[2], delimiter=',')


label_time = [int(os.path.splitext(x)[0]) for x in label_files]
print(label_time[0:5])
kinematics_time = kinematics_array[:,0]
new_kinematics_array = np.zeros((len(label_time),kinematics_array.shape[1]))
new_kinematics_array[:,0] = label_time
for j in range(1,kinematics_array.shape[1]):
    new_kinematics_array[:,j] = np.interp(label_time, kinematics_time, kinematics_array[:,j])

np.savetxt(kinematics_path[0:-4]+'_interpolated.csv', new_kinematics_array,delimiter=',')

