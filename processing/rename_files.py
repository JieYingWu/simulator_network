import os
import sys
import shutil

# Move a file by renaming it's path
path = sys.argv[1]
num = int(sys.argv[2])
for i in range(1,num+1):
    os.rename(path+'position'+str(i)+'.txt', path+'position'+"{:04d}".format(i)+'.txt')
