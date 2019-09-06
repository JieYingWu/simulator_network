#!/bin/bash


for i in {0..4}
do
#    rostopic echo -b ../../dataset/2019-08-08-Lego/data${i}.bag -p /ndi/Pointer/position_cartesian_current > ../../dataset/2019-08-08-Lego/data${i}_polaris.csv
#    rostopic echo -b ../../dataset/2019-08-08-Lego/data${i}.bag -p /dvrk/PSM3/position_cartesian_current > ../../dataset/2019-08-08-Lego/data${i}_robot_cartesian.csv
    python register_robot.py ../../dataset/2019-08-08-Lego/data${i}_robot_cartesian.csv
#    python preprocess_polaris.py ../../dataset/2019-08-08-Lego/data${i}_polaris.csv
done

