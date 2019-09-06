
## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##                  Export to PLY                  ##
#####################################################

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulationimport numpy as np
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import argparse for command-line options
import argparse
# Import os.path for file path manipulation
import os.path
# So wait for frame is called frequently enough
import threading

def write_ply(name, frame):
    depth = frame.get_depth_frame()
    color = frame.get_color_frame()
        
    # Generate the pointcloud and texture mappings
    points = pc.calculate(depth)
    print("Saving to " + name + "/"  + str(time) + ".ply...")
    points.export_to_ply(name + "/"  + str(time) + ".ply", color)


# Create object for parsing command-line options
parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                Remember to change the stream resolution, fps and format to match the recorded.")
# Add argument which takes path to a bag file as an input
parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
parser.add_argument("-o", "--output", type=str, help="Path to output directory")
parser.add_argument("-n", "--num_frames", type=int, help="Number of frames to read out")
# Parse the command line arguments to an object
args = parser.parse_args()
# Safety if no parameter have been given
if not args.input:
    print("No input paramater have been given.")
    print("For help type --help")
    exit()
# Check if the given file have bag extension
if os.path.splitext(args.input)[1] != ".bag":
    print("The given file is not of correct file format.")
    print("Only .bag files are accepted")
    exit()

name = args.output
while(len(os.listdir(name)) < args.num_frames):
    try:
        # Declare pointcloud object, for calculating pointclouds and texture mappings
        pc = rs.pointcloud()
        # We want the points object to be persistent so we can display the last cloud when a frame drops
        points = rs.points()

        # Declare RealSense pipeline, encapsulating the actual device and sensors
        pipe = rs.pipeline()

        # Create a config object
        config = rs.config()
        # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
        rs.config.enable_device_from_file(config, args.input)
        # Configure the pipeline to stream the depth stream
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)

        #Start streaming with default recommended configuration
        pipe.start(config)
        threads = list()
    
        for t in range(args.num_frames):
            # Wait for the next set of frames from the camera
            frames = pipe.wait_for_frames()
            time = frames.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)

            if not os.path.exists(name + "/"  + str(time) + ".ply"):        
                t = threading.Thread(target=write_ply, args=(name, frames,))
                threads.append(t)
                t.start()
    finally:
        for index, thread in enumerate(threads):
            thread.join()
        pipe.stop()
