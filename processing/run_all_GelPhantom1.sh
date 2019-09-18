## Steps
## Segment a point cloud without instrument
## Use that to register to model
## Use the registration result to transform the rest

#!/bin/bash

path=../../dataset/2019-09-07-GelPhantom1
frames=(1438 1058 2405 1517 1955 1748 1917 1534 1273 2158 1687 1689)

for i in {8..11} #..11} #{0..7}
do
#    rostopic echo -b $path/dvrk/data${i}.bag -p /dvrk/PSM3/position_cartesian_current > $path/dvrk/data${i}_robot_cartesian.csv
#    mkdir $path/camera/data${i}
#    python extract_point_clouds.py -i $path/camera/data${i}.bag -o $path/camera/data${i} -n ${frames[$i]}
#    python register_robot.py $path/dvrk/data${i}_robot_cartesian.csv
#    python interpolate_kinematics.py $path/dvrk/data${i}_robot_cartesian_processed.csv $path/camera/data${i}/
    #    mkdir $path/camera/data${i}_filtered
    rm $path/camera/data${i}_filtered/*
    python downsample_point_cloud.py $path/camera/data${i}/ $path/camera/data${i}_filtered/
done
