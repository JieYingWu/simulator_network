#!/bin/bash


for i in {0..11}
do
    python register_robot.py ../../dataset/2019-08-14-GelPhantom1/dvrk/data${i}_robot_cartesian.csv
    python interpolate_kinematics.py ../../dataset/2019-08-14-GelPhantom1/dvrk/data${i}_robot_cartesian_processed.csv ../../dataset/2019-08-14-GelPhantom1/camera/data${i}/
done

