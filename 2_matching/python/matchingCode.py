# Takes all of the cropped flakes from 2 cameras, sorts them by when the picture was taken 
# and attempts to match cropped flakes from the same source image across cameras
import os, sys
import time
import numpy as np
import glob
from utils import sortByDate, toOutput, findCorrespondIndices

cam1Path = '/mnt/snowdata/SMAS_IMAGES/SMAS_CAM1/cropped/'
cam2Path = '/mnt/snowdata/SMAS_IMAGES/SMAS_CAM2/cropped/'

#cam1Path = 'test/cam1/'
#cam2Path = 'test/cam2/'

start_time = time.time()

cam1Images = [f for f in os.listdir(cam1Path) if os.path.isfile(os.path.join(cam1Path, f))]
cam2Images = [f for f in os.listdir(cam2Path) if os.path.isfile(os.path.join(cam2Path, f))]

print(len(cam1Images), " flakes cropped from cam1")
print(len(cam2Images), " flakes cropped from cam2")

cam1ByDate = sortByDate(cam1Images)
cam2ByDate = sortByDate(cam2Images)

cam1Nums = cam1ByDate[:]
cam1Nums = np.asarray([n[5:11] for n in cam1Nums])
cam2Nums = cam2ByDate[:]
cam2Nums = np.asarray([n[5:11] for n in cam2Nums])
intersect = np.intersect1d(cam1Nums, cam2Nums)

out1 = []
out2 = []

for i in intersect:
    st1 = cam1Path + "Flake" + i + "*"
    cam1Flakes = sorted(glob.glob(st1))
    out1.append(toOutput(cam1Flakes))

    st2 = cam2Path + "Flake" + i + "*"
    cam2Flakes = sorted(glob.glob(st2))
    out2.append(toOutput(cam2Flakes))

    # print("Camera 1 set: ", out1)
    # print("Camera 2 set: ", out2)

# Fundamental matrix for the two camers being matched

#Matrix for cam 0 and 1
#F10 = array([[ 7.51923156e-08,  2.81755226e-07, -3.79207973e-04],
#       [ 2.25512084e-07, -4.67941341e-08, -1.48810814e-03],
#       [-4.38591473e-05,  1.81551149e-03,  1.00000000e+00]])

F12 = np.array([[ 2.21107922e-07, -6.04527898e-07,  5.62471057e-04],
       [-4.52700820e-07, -1.50986002e-07, -2.35488590e-03],
       [-7.58463124e-04,  5.45459685e-03,  1.00000000e+00]])
       
count = 0
indices1to2 = []
indices2to1 = []
for o1,o2 in zip(out1,out2):
    # print("Set",count)
    o1 = np.asarray(o1)
    o2 = np.asarray(o2)
    count+=1
#     o1 = np.asarray([o1[:,1],o1[:,0]]).T
#     o2 = np.asarray([o2[:,1],o2[:,0]]).T
    ind2,ind1 = findCorrespondIndices(o1,o2,F12,(1920,1200),(2448,2048))
    indices1to2.append(ind1)
    indices2to1.append(ind2)

pairedFiles = []
for fileList,i1,i2 in zip(intersect,indices1to2,indices2to1):
    print(i1,i2)
    st1 = cam1Path + "Flake" + fileList + "*"
    cam1Flakes = sorted(glob.glob(st1))
    st2 = cam2Path + "Flake" + fileList + "*"
    cam2Flakes = sorted(glob.glob(st2))

    for i in range(len(i1)):
        if(i1[i] >=0 ):
            if(i2[int(i1[i])] == i):
                pairedFiles.append([cam1Flakes[int(i1[i])],cam2Flakes[i]])

# This program will be run twice for two different pairs so make sure that the outputs are written to different files
print(pairedFiles)
f = open("pairedFlakeKey.txt", 'w')
f.write(pairedFiles)
f.close()

print("----execution took %s seconds----" % (time.time()-start_time)) 
