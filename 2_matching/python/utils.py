import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.spatial.transform import Rotation as R
import glob
import os
import random

# ==================================================================================
# Image matching utilities
# ==================================================================================

def readFile(path):
    f = open(path, 'r')
    data = f.read()
    return data.split('], [')

def findNth(s, ss, n):
    sep = s.split(ss, n)
    if len(sep) <= n:
        return -1
    return len(s) - len(sep[-1]) - len(ss)

def cleanList(l):
    out = []
    for i in range(0, len(l)):
        l1 = l[i][l[i].index('/home'):l[i].index('.png')+4]
        l2 = l[i][findNth(l[i], '/home', 2):findNth(l[i], '.png', 2)+4]

        out.append((l1, l2))
    return out

def compileList(list1, list2):
    out = []
    for l1 in enumerate(list1):
        for l2 in enumerate(list2):
            if l1[0] == l2[0]:
                out.append((l1[0], l1[1], l2[1]))
    return out

def getList(path1, path2, v=0):
    list1 = cleanList(readFile(path1))
    list2 = cleanList(readFile(path2))
    list3 = compileList(list1, list2)

    if v == 1:
        print("Number of pairs found between first pair of cameras: ", len(list1))
        print("Number of pairs found between second pair of cameras: ", len(list2))
        print("Number of triplets fround across all three cameras: ", len(list3))
        print(list3)
    return list3

def sortByFlake(l):
    return sorted(l)

def getDate(flake):
    flake = flake[24:]
    date = "{:02d}".format(int(flake[:flake.index('-')])) # Month
    flake = flake[flake.index('-')+1:]
    date += "{:02d}".format(int(flake[:flake.index('-')])) # Day
    flake = flake[flake.index('-')+1:]
    date += "{:02d}".format(int(flake[:flake.index('-')])) # Hour
    flake = flake[flake.index('-')+1:]
    date += "{:02d}".format(int(flake[:flake.index('-')])) # Minute
    flake = flake[flake.index('-')+1:]
    date += "{:02d}".format(int(flake[:flake.index('-')])) # Second
    flake = flake[flake.index('-')+1:]
    date += "{:03d}".format(int(flake[:flake.index('X')])) # Millisecond
    flake = flake[flake.index('X')+1:]
    date += "{:02d}".format(int(flake[flake.index('d')+1:flake.index('.')])) # Cropped Number
    return date

def sortByDate(l):
    return sorted(l, key=getDate)

def getIndex(l, num):
    i = 1
    while(True):
        if num not in l[i]:
            return i
        i += 1
        if i >= 101:
            print("\n\tWarning: More than 100 flakes found from one source image\n")

def toOutput(camSet):
    out = []
    for f in camSet:
        out.append((int(f[f.index('X')+2:f.index('Y')]), int(f[f.index('Y')+2:f.index('r')-1])))
    out = np.asarray(out)
    return out

# ======================================================================================================
# Fundamental Matrix Utilities
# ======================================================================================================

#homogeneous transformation multiplication
def computeHomo(e0,e1):
    e0 = np.asarray(e0)
    e1 = np.asarray(e1)
    e0 = np.vstack((e0,np.zeros((1,4))))
    e1 = np.vstack((e1,np.zeros((1,4))))
    e0[3,3] = 1
    e1[3,3] = 1
    return np.matmul(e0,e1)

#initial Kabsch Algorithm using SVD on Covariance to find rotation and translation
def kabsch_umeyama(A, B):
    assert A.shape == B.shape
    n, m = A.shape

    EA = np.mean(A, axis=0)
    EB = np.mean(B, axis=0)
    VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)

    H = ((A - EA).T @ (B - EB)) / n
    U, D, VT = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
    S = np.diag([1] * (m - 1) + [d])

    R = U @ S @ VT
    c = VarA / np.trace(np.diag(D) @ S)
    t = EA - c * R @ EB

    return R, c, t

# find Extrinsics using one of 2 methods wrapper
def findExtrinsics(mtx0,dist0,mtx1,dist1,objpoints0,imgpoints0,objpoints1,imgpoints1,rvecs0,tvecs0,rvecs1,tvecs1,method=2):
    if method==1:
        imgPointsSelected0 = []
        imgPointsSelected1 = []
        objPointsSelected = []
        for i in range(len(objpoints0)):
            o0 = np.asarray(objpoints0[i])
            i0 = np.asarray(imgpoints0[i])
            o1 = np.asarray(objpoints1[i])
            i1 = np.asarray(imgpoints1[i])
            o0set = set([tuple(x) for x in o0])
            o1set = set([tuple(x) for x in o1])
            objPt = np.array([x for x in o0set & o1set])
            objPts = []
            imgs0 = []
            imgs1 = []
            for o in objPt:
                ind0=np.where(np.all(o0==o,axis=1))[0][0]
                ind1=np.where(np.all(o1==o,axis=1))[0][0]
                objPts.append(o.reshape((1,3)))
                imgs0.append(i0[ind0,:])
                imgs1.append(i1[ind1,:])
            ob = np.asarray(objPts)
            print(ob.shape)
            if(len(ob)>2):
                ob = ob.reshape((ob.shape[0],ob.shape[2]))
                objPointsSelected.append(ob)
                imgPointsSelected0.append(np.asarray(imgs0))
                imgPointsSelected1.append(np.asarray(imgs1))

        (rpe,mtx0,dist0,mtx1,dist1,R,T,E,F) = cv.stereoCalibrate(objPointsSelected,imgPointsSelected0,imgPointsSelected1,mtx0,dist0,mtx1,dist1,(2448,2048),flags=cv.CALIB_FIX_INTRINSIC,criteria=(cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS,1000000000,1e-14))
        rvecs = cv.Rodrigues(np.asarray(R))[0]
        return R,T
    elif method==2:
        pts0=[]
        pts1=[]
        for i in range(len(rvecs0)):
            ext0 = np.hstack((cv.Rodrigues(np.asarray(rvecs0[i],dtype=np.float32))[0],np.asarray(tvecs0[i]).reshape((3,1))))
            ext0 = np.vstack((ext0,np.zeros((1,4))))
            ext0[3,3] = 1
            ext1 = np.hstack((cv.Rodrigues(np.asarray(rvecs1[i],dtype=np.float32))[0],np.asarray(tvecs1[i]).reshape((3,1))))
            ext1 = np.vstack((ext1,np.zeros((1,4))))
            mainVec = np.array([[1],[1],[1],[1]])
            ext1[3,3] = 1
            pts0.append(np.matmul(ext0,mainVec))
            pts1.append(np.matmul(ext1,mainVec))
        pts0 = (np.asarray(pts0))
        pts0 = np.asarray(pts0).reshape((pts0.shape[0],pts0.shape[1]))[:,:3]
        pts1 = (np.asarray(pts1))
        pts1 = np.asarray(pts1).reshape((pts1.shape[0],pts1.shape[1]))[:,:3]
        R01,c,t = kabsch_umeyama(pts0,pts1)
        return R01,t

# Kabsch extrinsics finding wrapper
def findExtrinsics2(rvecs0,rvecs1,tvecs0,tvecs1):
    pts0=[]
    pts1=[]
    for i in range(len(rvecs0)):
        ext0 = np.hstack((cv.Rodrigues(np.asarray(rvecs0[i],dtype=np.float32))[0],np.asarray(tvecs0[i]).reshape((3,1))))
        ext0 = np.vstack((ext0,np.zeros((1,4))))
        ext0[3,3] = 1
        ext1 = np.hstack((cv.Rodrigues(np.asarray(rvecs1[i],dtype=np.float32))[0],np.asarray(tvecs1[i]).reshape((3,1))))
        ext1 = np.vstack((ext1,np.zeros((1,4))))
        mainVec = np.array([[1],[1],[1],[1]])
        ext1[3,3] = 1
        pts0.append(np.matmul(ext0,mainVec))
        pts1.append(np.matmul(ext1,mainVec))
    pts0 = (np.asarray(pts0))
    pts0 = np.asarray(pts0).reshape((pts0.shape[0],pts0.shape[1]))[:,:3]
    pts1 = (np.asarray(pts1))
    pts1 = np.asarray(pts1).reshape((pts1.shape[0],pts1.shape[1]))[:,:3]
    R01,c,t = kabsch_umeyama(pts0,pts1)
    rvec = cv.Rodrigues(np.asarray(R01))[0]*180/np.pi
    # rvecNew = 
    return rvec.flatten(),t
    

# Calibration 1
def extractCalibrationMatrices(location0_wPt,location0_iPt,location1_wPt,location1_iPt,imageLocation0,imageLocation1,debug=True):
    objpoints0 = []
    imgpoints0 = []
    objpoints1 = []
    imgpoints1 = []
    
    if isinstance(location0_wPt, str):
        print("sorted")
        w0files = sorted(glob.glob(location0_wPt))
        i0files = sorted(glob.glob(location0_iPt))
        w1files = sorted(glob.glob(location1_wPt))
        i1files = sorted(glob.glob(location1_iPt))
    else:
        w0files = location0_wPt
        i0files = location0_iPt
        w1files = location1_wPt
        i1files = location1_iPt
    
    print(len(w0files),len(i0files),len(w1files),len(i1files))
    for i in range(len(w0files)):
        if os.stat(w0files[i]).st_size >20 and os.stat(i0files[i]).st_size >90 and os.stat(w1files[i]).st_size >20 and os.stat(i1files[i]).st_size >90:
            w_Pt = np.genfromtxt(w0files[i],delimiter=',')
            i_Pt = np.genfromtxt(i0files[i],delimiter=',')
            w_Pt = np.hstack((w_Pt,np.zeros((len(w_Pt),1))))
            objpoints0.append(w_Pt.astype(np.float32))
            imgpoints0.append(i_Pt.astype(np.float32))
            
            w_Pt = np.genfromtxt(w1files[i],delimiter=',')
            i_Pt = np.genfromtxt(i1files[i],delimiter=',')
            w_Pt = np.hstack((w_Pt,np.zeros((len(w_Pt),1))))
            objpoints1.append(w_Pt.astype(np.float32))
            imgpoints1.append(i_Pt.astype(np.float32))

    if(cv.imread(imageLocation0).shape[2] == 3):
        im0 = cv.cvtColor(cv.imread(imageLocation0),cv.COLOR_BGR2GRAY)
    else:
        im0 = cv.imread(imageLocation0)
    if(cv.imread(imageLocation1).shape[2] == 3):
        im1 = cv.cvtColor(cv.imread(imageLocation1),cv.COLOR_BGR2GRAY)
    else:
        im1 = cv.imread(imageLocation1)
#     print(len(imgpoints0),len(imgpoints1))

    flags=(cv.CALIB_ZERO_TANGENT_DIST+cv.CALIB_FIX_K3)
    criteria=(cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS,10000,1e-8)
    (ret0,mtx0,dist0,rvecs0,tvecs0) = cv.calibrateCamera(objpoints0,imgpoints0,im0.shape[::-1],None,None,flags=flags,criteria=criteria)
    (ret1,mtx1,dist1,rvecs1,tvecs1) = cv.calibrateCamera(objpoints1,imgpoints1,im1.shape[::-1],None,None,flags=flags,criteria=criteria)
#     print((np.asarray(rvecs0)))
#     mean_error = 0
    me0 = 0
    me1 = 0
    for i in range(len(objpoints0)):
        proj0, _ = cv.projectPoints(objpoints0[i], rvecs0[i], tvecs0[i], mtx0, dist0)
        proj1, _ = cv.projectPoints(objpoints1[i], rvecs1[i], tvecs1[i], mtx1, dist1)
        proj0 = proj0.reshape((proj0.shape[0],proj0.shape[2]))
        proj1 = proj1.reshape((proj1.shape[0],proj1.shape[2]))
        error0 = cv.norm(imgpoints0[i], proj0, cv.NORM_L2)/len(proj0)
        error1 = cv.norm(imgpoints1[i], proj1, cv.NORM_L2)/len(proj1)
        me0 += error0
        me1 += error1
        if debug:
            print(i0files[i],"err0: ",error0,"err1: ", error1)
#     print( "total error: {}".format(mean_error/len(objpoints)) )
    print(len(objpoints0))
    me0 = me0/len(objpoints0)
    me1 = me1/len(objpoints1)
#     print("rpe0 = ",me0)
#     print("rpe1 = ",me1)
    R,T = findExtrinsics2(rvecs0,rvecs1,tvecs0,tvecs1)
    # rvec = cv.Rodrigues(np.asarray(R))[0]*180/np.pi
    
    return me0,me1,mtx0,dist0,mtx1,dist1,R,T.flatten()

def extractCalibrationMatrices2(location0_wPt,location0_iPt,location1_wPt,location1_iPt,imageLocation0,imageLocation1,debug=True,sameCam=False):
    objpoints0 = []
    imgpoints0 = []
    objpoints1 = []
    imgpoints1 = []
    
    if isinstance(location0_wPt, str):
        print("sorted")
        w0files = sorted(glob.glob(location0_wPt))
        i0files = sorted(glob.glob(location0_iPt))
        w1files = sorted(glob.glob(location1_wPt))
        i1files = sorted(glob.glob(location1_iPt))
    else:
        w0files = location0_wPt
        i0files = location0_iPt
        w1files = location1_wPt
        i1files = location1_iPt
    
    print(len(w0files),len(i0files),len(w1files),len(i1files))
    for i in range(len(w0files)):
        if os.stat(w0files[i]).st_size >20 and os.stat(i0files[i]).st_size >90 and os.stat(w1files[i]).st_size >20 and os.stat(i1files[i]).st_size >90:
            w_Pt = np.genfromtxt(w0files[i],delimiter=',')
            i_Pt = np.genfromtxt(i0files[i],delimiter=',')
            w_Pt = np.hstack((w_Pt,np.zeros((len(w_Pt),1))))
            objpoints0.append(w_Pt.astype(np.float32))
            imgpoints0.append(i_Pt.astype(np.float32))
            
            w_Pt = np.genfromtxt(w1files[i],delimiter=',')
            i_Pt = np.genfromtxt(i1files[i],delimiter=',')
            w_Pt = np.hstack((w_Pt,np.zeros((len(w_Pt),1))))
            objpoints1.append(w_Pt.astype(np.float32))
            imgpoints1.append(i_Pt.astype(np.float32))

    if(cv.imread(imageLocation0).shape[2] == 3):
        im0 = cv.cvtColor(cv.imread(imageLocation0),cv.COLOR_BGR2GRAY)
    else:
        im0 = cv.imread(imageLocation0)
    if(cv.imread(imageLocation1).shape[2] == 3):
        im1 = cv.cvtColor(cv.imread(imageLocation1),cv.COLOR_BGR2GRAY)
    else:
        im1 = cv.imread(imageLocation1)
#     print(len(imgpoints0),len(imgpoints1))

    flags=(cv.CALIB_ZERO_TANGENT_DIST+cv.CALIB_FIX_K3)
    criteria=(cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS,10000,1e-8)
    (ret0,mtx0,dist0,rvecs0,tvecs0) = cv.calibrateCamera(objpoints0,imgpoints0,im0.shape[::-1],None,None,flags=flags,criteria=criteria)
    (ret1,mtx1,dist1,rvecs1,tvecs1) = cv.calibrateCamera(objpoints1,imgpoints1,im1.shape[::-1],None,None,flags=flags,criteria=criteria)
#     print((np.asarray(rvecs0)))
#     mean_error = 0
    me0 = 0
    me1 = 0
    for i in range(len(objpoints0)):
        proj0, _ = cv.projectPoints(objpoints0[i], rvecs0[i], tvecs0[i], mtx0, dist0)
        proj1, _ = cv.projectPoints(objpoints1[i], rvecs1[i], tvecs1[i], mtx1, dist1)
        proj0 = proj0.reshape((proj0.shape[0],proj0.shape[2]))
        proj1 = proj1.reshape((proj1.shape[0],proj1.shape[2]))
        error0 = cv.norm(imgpoints0[i], proj0, cv.NORM_L2)/len(proj0)
        error1 = cv.norm(imgpoints1[i], proj1, cv.NORM_L2)/len(proj1)
        me0 += error0
        me1 += error1
        if debug:
            print(i0files[i],"err0: ",error0,"err1: ", error1)
#     print( "total error: {}".format(mean_error/len(objpoints)) )
    objpoints = []
    imgpoints0_sel = []
    imgpoints1_sel = []
    for i in range(len(objpoints0)):
        A = np.array(objpoints0[i])
        B = np.array(objpoints1[i])
        aset = set([tuple(x) for x in A])
        bset = set([tuple(x) for x in B])
        intersect = np.array([x for x in aset & bset])
        indA = []
        indB = []
        for ix in intersect:
            indA.append(np.where((A == ix).all(axis=1)))
            indB.append(np.where((B == ix).all(axis=1)))
        indA = np.asarray(indA).flatten()
        indB = np.asarray(indB).flatten()
        objpoints.append(intersect)
        # print(imgpoints0[i].shape)
        # print(indA)
        imgpoints0_sel.append(imgpoints0[i][indA,:])
        imgpoints1_sel.append(imgpoints1[i][indB,:])

    # print(imgpoints0_sel)
    me0 = me0/len(objpoints0)
    me1 = me1/len(objpoints1)
#     print("rpe0 = ",me0)
#     print("rpe1 = ",me1)
    # R,T = findExtrinsics2(rvecs0,rvecs1,tvecs0,tvecs1)
    if sameCam:
        res = cv.stereoCalibrate(objpoints,imgpoints0_sel,imgpoints1_sel,mtx0,dist0,mtx1,dist1,im0.shape[::-1])#,flags=cv.CALIB_FIX_INTRINSIC)
    else:
        res = cv.stereoCalibrate(objpoints,imgpoints0_sel,imgpoints1_sel,mtx0,dist0,mtx1,dist1,im0.shape[::-1],flags=cv.CALIB_FIX_INTRINSIC)
    # print(res)
    rvec = cv.Rodrigues(np.asarray(res[5]))[0]*180/np.pi
    t = res[6]
    #temp
    mtx0 = res[1]
    dist0 = res[2]
    mtx1 = res[3]
    dist1 = res[4]
    E = res[7]
    F = res[8]
    return me0,me1,mtx0,dist0,mtx1,dist1,rvec.flatten(),t.flatten(),E,F

def extractCalibrationMatricesMin(indF,indT,location0_wPt,location0_iPt,location1_wPt,location1_iPt,imageLocation0,imageLocation1):
    objpoints0 = []
    imgpoints0 = []
    objpoints1 = []
    imgpoints1 = []
    supMat_1 = np.zeros((0,6))
    optMat_1 = np.zeros((0,6)) 
    
    w0files = sorted(glob.glob(location0_wPt))
    i0files = sorted(glob.glob(location0_iPt))
    w1files = sorted(glob.glob(location1_wPt))
    i1files = sorted(glob.glob(location1_iPt))
    print(location0_wPt)
    for i in range(len(w0files)):
        if os.stat(w0files[i]).st_size >25 and os.stat(i0files[i]).st_size >100 and os.stat(w1files[i]).st_size >25 and os.stat(i1files[i]).st_size >100:
            w_Pt = np.genfromtxt(w0files[i],delimiter=',')
            i_Pt = np.genfromtxt(i0files[i],delimiter=',')
            w_Pt = np.hstack((w_Pt,np.zeros((len(w_Pt),1))))
            objpoints0.append(w_Pt.astype(np.float32))
            imgpoints0.append(i_Pt.astype(np.float32))
            
            supMat_1_0 = np.array([0,0,0,0,i0files[i],w0files[i]])
            supMat_1_0[indF] = 1
            supMat_1_0[indT+2] = 1
            supMat_1 = np.append(supMat_1,supMat_1_0.reshape((1,6)),axis=0)
            
            w_Pt = np.genfromtxt(w1files[i],delimiter=',')
            i_Pt = np.genfromtxt(i1files[i],delimiter=',')
            w_Pt = np.hstack((w_Pt,np.zeros((len(w_Pt),1))))
            objpoints1.append(w_Pt.astype(np.float32))
            imgpoints1.append(i_Pt.astype(np.float32))
            
            supMat_1_1 = np.array([0,0,0,0,i1files[i],w1files[i]])
            supMat_1_1[indT] = 1
            supMat_1_1[indF+2] = 1
            supMat_1 = np.append(supMat_1,supMat_1_1.reshape((1,6)),axis=0)
    
    if(cv.imread(imageLocation0).shape[2] == 3):
        im0 = cv.cvtColor(cv.imread(imageLocation0),cv.COLOR_BGR2GRAY)
    else:
        im0 = cv.imread(imageLocation0)
    if(cv.imread(imageLocation1).shape[2] == 3):
        im1 = cv.cvtColor(cv.imread(imageLocation1),cv.COLOR_BGR2GRAY)
    else:
        im1 = cv.imread(imageLocation1)
        
    flags=(cv.CALIB_ZERO_TANGENT_DIST+cv.CALIB_FIX_K3)
    criteria=(cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS,10000,1e-12)
    (ret0,mtx0,dist0,rvecs0,tvecs0) = cv.calibrateCamera(objpoints0,imgpoints0,im0.shape[::-1],None,None,flags=flags,criteria=criteria)
    (ret1,mtx1,dist1,rvecs1,tvecs1) = cv.calibrateCamera(objpoints1,imgpoints1,im1.shape[::-1],None,None,flags=flags,criteria=criteria)
    
    # print(mtx0)
    # print(mtx1)
    # print(dist0)
    # print(dist1)
    me0 = 0
    me1 = 0
    for i in range(len(objpoints0)):
        proj0, _ = cv.projectPoints(objpoints0[i], rvecs0[i], tvecs0[i], mtx0, dist0)
        proj1, _ = cv.projectPoints(objpoints1[i], rvecs1[i], tvecs1[i], mtx1, dist1)
        proj0 = proj0.reshape((proj0.shape[0],proj0.shape[2]))
        proj1 = proj1.reshape((proj1.shape[0],proj1.shape[2]))
        error0 = cv.norm(imgpoints0[i], proj0, cv.NORM_L2)/len(proj0)
        error1 = cv.norm(imgpoints1[i], proj1, cv.NORM_L2)/len(proj1)
        me0 += error0
        me1 += error1
        if False:
            print(i0files[i],"err0: ",error0,"err1: ", error1)
            
    for i in range(len(rvecs0)):
        rv = np.asarray(rvecs0[i])
        tv = np.asarray(tvecs0[i])
        vec = np.array([rv[0],rv[1],rv[2],tv[0],tv[1],tv[2]])
        # print(vec.flatten())
        optMat_1=np.append(optMat_1,vec.reshape((1,6)),axis=0)
        rv1 = np.asarray(rvecs1[i])
        tv1 = np.asarray(tvecs1[i])
        vec1 = np.array([rv1[0],rv1[1],rv1[2],tv1[0],tv1[1],tv1[2]])
        # print(vec1.flatten())
        optMat_1=np.append(optMat_1,vec1.reshape((1,6)),axis=0)
    
    intrinsics0 = np.asarray(mtx0)
    intrinsics1 = np.asarray(mtx1)
    distortion_coeffs0 = np.asarray(dist0)
    distortion_coeffs1 = np.asarray(dist1)
    
    return intrinsics0,intrinsics1,distortion_coeffs0,distortion_coeffs1,optMat_1,supMat_1

#cv method for calculating RPE
def calculateReprojectionError(objpoints,mtx,dist,rvecs,tvecs):
    for i in range(len(objpoints)):
        proj, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        proj = proj.reshape((proj.shape[0],proj.shape[2]))
        error = cv.norm(imgpoints[i], proj, cv.NORM_L2)/len(proj)
        me += error
        if debug:
            print(i0files[i],"err: ",error)
    return me/len(objpoints)


# bootstrapping images for comparing calibration accuracy
def bootStrapping(i0,i1,o0,o1,size=3,amount=100):
    idx = []
    i0Bootstrapped = []
    i1Bootstrapped = []
    o0Bootstrapped = []
    o1Bootstrapped = []
    inx = range(0,len(i0)-1)
    print(len(inx))

    for i in range(amount):
        idx.append(random.sample(inx, size))
    
    for id in idx:
        id = np.asarray(id,dtype=int)
        i0Bootstrapped.append(i0[id])
        i1Bootstrapped.append(i1[id])
        o0Bootstrapped.append(o0[id])
        o1Bootstrapped.append(o1[id])

    return i0Bootstrapped,i1Bootstrapped,o0Bootstrapped,o1Bootstrapped,idx

# printing files of index
def printNotWorkingGroup(files,id):
    files=np.asarray(files)
    return files[id]
    

# bootstrapping~ish method for only removing one image at a time sequentially
def bootStrappingSequential(i0,i1,o0,o1):
    idx = []
    i0Bootstrapped = []
    i1Bootstrapped = []
    o0Bootstrapped = []
    o1Bootstrapped = []
    inx = range(0,len(i0)-1)
    print(len(inx))

    for i in range(len(inx)):
        idx.append(np.setxor1d(inx,np.array([i])))
    
    for id in idx:
        id = np.asarray(id,dtype=int)
        print(id)
        i0Bootstrapped.append(i0[id])
        i1Bootstrapped.append(i1[id])
        o0Bootstrapped.append(o0[id])
        o1Bootstrapped.append(o1[id])

    return i0Bootstrapped,i1Bootstrapped,o0Bootstrapped,o1Bootstrapped,idx

# matching method
def findCorrespondIndices(pts1,pts2,F,img1Size,img2Size,distanceThresh=15,debug=False):
    #create epippolar lines
    lines2 = cv.computeCorrespondEpilines(np.array([pts1]), 1, F)
    lines2 = lines2.reshape(-1,3)
    lines1 = cv.computeCorrespondEpilines(np.array([pts2]), 2, F)
    lines1 = lines1.reshape(-1,3)

    # debug. if debug=False, imgSizes are ignored
    if debug:
        #draw
        img2, img1 = drawlines(img2Size,img1Size,lines2,pts2,pts1)
        img3, img4 = drawlines(img1Size,img2Size,lines1,pts1,pts2)
        plt.figure(figsize=(30,30))
        plt.subplot(2,2,1)
        plt.imshow(img1.astype(np.uint8))
        plt.title("Cam 1 to 2")
        plt.subplot(2,2,2)
        plt.imshow(img2.astype(np.uint8))
        plt.subplot(2,2,3)
        plt.imshow(img3.astype(np.uint8))
        plt.title("Cam 2 to 1")
        plt.subplot(2,2,4)
        plt.imshow(img4.astype(np.uint8))
    #distance matching
    matchingIndices2to1 = np.zeros((pts1.shape[0]))-99
    for j,line in enumerate(lines2):
        for i,point in enumerate(pts2):
            dist = shortestDist(point,line)
            if(dist<distanceThresh):
                matchingIndices2to1[j] = i

    matchingIndices1to2 = np.zeros((pts2.shape[0]))-99
    for j,line in enumerate(lines1):
        for i,point in enumerate(pts1):
            dist = shortestDist(point,line)
            if(dist<distanceThresh):
                matchingIndices1to2[j] = i
    
    return matchingIndices2to1, matchingIndices1to2

# matching debug helper for drawing 4 images for epipolar geometry
def drawlines(img1Size,img2Size,lines,pts1,pts2,thickness=3,size=10):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    # img1 = Image.fromarray(img1.astype(np.uint8))
    # img2 = Image.fromarray(img2.astype(np.uint8))

    img1 = np.zeros((img1Size[1],img1Size[0],3))
    img2 = np.zeros((img2Size[1],img2Size[0],3))
    # for i in range(3):
    #     img1[:,:,i] = n1
    #     img2[:,:,i] = n2
    # img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    # img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    r,c,_ = img1.shape
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1),  color,thickness)
        loc1 = (int(pt1[0]),int(pt1[1]))
        loc2 = (int(pt2[0]),int(pt2[1]))
        img1 = cv.circle(img1,loc1,size,[255,255,255],-1)
        img2 = cv.circle(img2,loc2,size,color,-1)
    return img1,img2

# distance calculator from point to line helper for matching
def shortestDist(point,line):
    a = line[0]
    b = line[1]
    c = line[2]
    xp = point[0]
    yp = point[1]

    x = 1/(a**2+b**2)*(b**2*xp-a*b*yp-a*c)
    y = -a*x/b-c/b
    
    return np.sqrt((yp-y)**2+(xp-x)**2)
