## Steps
## Segment a point cloud without instrument
## Use that to register to model
## Use the registration result to transform the rest

#!/bin/bash

path=../../dataset/2019-10-09-GelPhantom1
frames=(3794 2410 2007 1926 1622 1982 1573 1987 2137 1890 1363 1394)
# calibration has 1626 msgs for 2019-10-09-GelPhantom1

for i in {0..11} #..11} #{0..7}
do
#    rostopic echo -b $path/dvrk/data${i}.bag -p /dvrk/PSM3/position_cartesian_current > $path/dvrk/data${i}_robot_cartesian.csv
#    mkdir $path/camera/data${i}
#    python extract_point_clouds.py -i $path/camera/data${i}.bag -o $path/camera/data${i} -n ${frames[$i]}
#    python register_robot.py $path/dvrk/data${i}_robot_cartesian.csv
#    python interpolate_kinematics.py $path/dvrk/data${i}_robot_cartesian_processed.csv $path/camera/data${i}/
    mkdir $path/camera/data${i}_filtered
    rm $path/camera/data${i}_filtered/*
    python downsample_point_cloud.py $path/camera/data${i}/ $path/camera/data${i}_filtered/
done
