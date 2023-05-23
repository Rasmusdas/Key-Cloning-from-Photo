import subprocess
import sys

# implement pip as a subprocess:
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
# 'opencv-python'])

# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
# 'numpy'])

# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
# 'scikit-image'])

# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
# 'matplotlib'])


import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math
from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                white_tophat)
import os
from skimage.morphology import disk  # noqa



results = []

def gaussian1DKernel(sigma):
    x = np.arange(-5.*sigma,5.*sigma)
    
    g = 1/np.sqrt(2.*math.pi*sigma**2.)*np.exp(-x**2./(2.*sigma**2.))
    
    gd = g * -x/(sigma**2.)
    
    return np.array([g]),np.array([gd])
    
def gaussianSmoothing(im, sigma):
    g, gd = gaussian1DKernel(sigma)
    i_b = cv.filter2D(src=im,ddepth=-1,kernel=g*g.T)
    i_y = cv.filter2D(src=im,ddepth=-1,kernel=g*gd.T)
    i_x = cv.filter2D(src=im,ddepth=-1,kernel=g.T*gd)
    return i_b, i_x, i_y

def smoothedHessian(im, sigma, epsilon):
    im, ix,iy = gaussianSmoothing(im,sigma)
    g, gd = gaussian1DKernel(epsilon)

    h1 = gaussianSmoothing(ix**2,epsilon)[0]
    h2 = gaussianSmoothing(ix*iy,epsilon)[0]
    h4 = gaussianSmoothing(iy**2,epsilon)[0]
    
    return np.array([[h1,h2],[h2,h4]])
    
def harrisMeasure(im,sigma,epsilon,k):
    C = smoothedHessian(im,sigma,epsilon)

    a = C[0,0]
    b = C[1,1]
    c = C[0,1]
    
    C = a*b-c**2-k*(a+b)**2
    
    return a*b-c**2-k*(a+b)**2

def displayImages(imgs,cmaps = []):
    col = math.ceil(math.sqrt(len(imgs)))
    row = round(math.sqrt(len(imgs)))
    fig = plt.figure(figsize=(8,8))
    for i in range(len(imgs)):
        fig.add_subplot(col,row,i+1)
        map = "gray"
        if len(cmaps) > i:
            if cmaps[i] == "":
                plt.imshow(imgs[i])
            else:
                map = cmaps[i]
                plt.imshow(imgs[i],cmap=map)
        
        


def detect_key(img,its,kernelRadius = 4):
    orgim = img
    
    # Grayscale and Firstblur
    img = cv.GaussianBlur(img,(9,9),3)

    # Edge detection with Prewitt Filter
    kernel = np.array([[-1 for j in range(kernelRadius*2+1)] for i in range(kernelRadius*2+1)])
    
    kernel[kernelRadius,kernelRadius] = (kernelRadius*2+1)**2-1
    keyEdge = cv.filter2D(img,-1,kernel)
    
    # kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    # kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    # img_prewittx = cv.filter2D(img, -1, kernelx)
    # img_prewitty = cv.filter2D(img, -1, kernely)
    
    

    # Noise Removal Iterations
    tH = 240
    for i in range(its):
        keyEdge[keyEdge < tH] = 0 
        keyEdge = cv.medianBlur(keyEdge,7)
    
    # End noise removal
    if its > 0:
        keyEdge[keyEdge < tH] = 0 
    
    # Get bounding box
    ind = np.asarray(keyEdge >=tH).nonzero()
    minx = ind[0][0]
    maxx = ind[0][-1]


    return orgim[minx:maxx,np.min(ind[1]):np.max(ind[1])],keyEdge

def threshold_key(img,min,max):
    cropped_gray = img
    cropped_gray[np.logical_and(cropped_gray > min, cropped_gray < max)] = 0
    
    cropped_gray = cv.medianBlur(cropped_gray,3)
    
    return cropped_gray

    
def get_contour(img, draw=False):
    img = np.copy(img)
    croppier_gray = img[0:img.shape[0]//2,img.shape[1]//2:-1]
    croppier_gray[croppier_gray > 20] = 255
    croppier_gray[croppier_gray <= 20] = 0
    croppier_gray[np.cumsum(croppier_gray,axis=0) > 0] = 255
    
    contours, _ = cv.findContours(croppier_gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    if draw:
        croppier_gray = cv.cvtColor(croppier_gray,cv.COLOR_GRAY2RGB)
        cv.drawContours(croppier_gray, contours, -1, (255,0,0), 1)

    return contours, croppier_gray

def distance_to_pixels(img,thickness):
    dist_kernel = np.ones((thickness,thickness))
    return cv.filter2D(img,-1,dist_kernel)

def draw_bounding_boxes(img,boxes):
    new_img = np.copy(img)
    
    for i in range(len(boxes)):
        x1 = boxes[i][0]
        y1 = boxes[i][1]

        x2 = boxes[i][2]
        y2 = boxes[i][3]
        
        cv.rectangle(new_img,(x1,y1),(x2+x1,y2+y1),(255,0,0),3)
        
    return new_img


def fill_key(img):
    img = np.copy(img)
    
    img = img.T

    img[img > 0] = 255
    
    for i in range(img.shape[0]):
        min_val = np.argmax(img[i])
        max_val = np.argmax(np.flip(img[i]))
        
        if min_val == 0: continue
        if abs(min_val-len(img[i])+max_val) < 35:
            min_val = np.argmax(img[i-1])
            max_val = np.argmax(np.flip(img[i-1]))
        img[i,min_val:len(img[i])-max_val] = 255
        
    img = img.T
    return img

def small_removed(img,stats):
    

    new_stats = stats
    avg = np.average(new_stats[:,4])
    
    new_stats = new_stats[new_stats[:,4] <= avg//4]
    
    
    fixed_img = np.copy(img)
    
    for new_stat in new_stats:
        fixed_img[new_stat[1]:new_stat[1]+new_stat[3],new_stat[0]:new_stat[0]+new_stat[2]] = 0
        
        
    return fixed_img
    
def fix_key_holes(img):
    img = np.copy(img)
    img = img.T
    allowed_delta = 1
    last_val = np.argmax(img[0])
    changed_delta = 0
    for i in range(img.shape[0]):
            new_val = np.argmax(img[i])

            new_delta = new_val-last_val
        
            if abs(new_delta) > 1:
                changed_delta = new_delta/abs(new_delta)
                img[i][0:int(last_val+changed_delta)] = 0
                img[i][int(last_val+changed_delta):img.shape[1]-1] = 255
                new_val = np.argmax(img[i])

            last_val = new_val

    return img.T

def shadow_magic(img):
    rgb_planes = cv.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv.medianBlur(dilated_img, 21)
        diff_img = 255 - cv.absdiff(plane, bg_img)
        norm_img = cv.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
        
    result = cv.merge(result_planes)
    result_norm = cv.merge(result_norm_planes)
    
    return result_norm

def rotate(image, angle, center = None, scale = 1.0):
    h, w = image.shape[:2]

    back = np.copy(cv.resize(image,(max(w, h),max(w, h))))
    back[:] = 0
    hh, ww = back.shape[:2]
    
    yoff = round((hh-h)/2)
    xoff = round((ww-w)/2)

    result = back.copy()
    result[yoff:yoff+h, xoff:xoff+w] = image

    if center is None:
        center = (max(w, h) / 2, max(w, h) / 2)

    M = cv.getRotationMatrix2D(center, angle, scale)
    rotated = cv.warpAffine(result, M, (max(w, h),max(w, h)))

    return rotated
    
def get_best_rotation(img,lineDetectingThreshold):
    lines = cv.HoughLines(img, 1, math.pi/180, lineDetectingThreshold)

    lines = sorted(lines, key=lambda x: x[0][0], reverse=True)

    line = lines[:10]
    
    theta = np.average(np.array(line)[:,:,1])
    
    rho = np.average(np.array(line)[:,:,0])

    degree = 90-180*theta/math.pi
    
    return rho,degree

def make_line(p1,p2):
    l = np.cross(p1,p2)
    
    return l/np.linalg.norm(l)

def get_inliers(l,p,t):
    dists = l@piInv(p.T)
    dists = abs(dists)    
    return p.T[:,(dists < t)[0]], dists[dists <t].shape
    
def make_random_line(p):
    n1 = math.floor(np.random.random() * p.shape[0])
    n2 = math.floor(np.random.random() * p.shape[0])
    
    while n1 == n2:
        n2 = math.floor(np.random.random() * p.shape[0])
        
    p1 = np.array([p[n1]]).T
    p2 = np.array([p[n2]]).T    
    
    return make_line(piInv(p1).T,piInv(p2).T)

def pca_line(x): #assumes x is a (2 x n) array of points
    d = np.cov(x)[:, 0]
    d /= np.linalg.norm(d)
    l = [d[1], -d[0]]
    l.append(-(l@x.mean(1)))
    return l
def piInv(p):
    ph = np.vstack((p,np.ones(p.shape[1])))
    return ph

def pi(ph):
    p = ph[:-1]/ph[-1]
    return p

def homogeneous_line_slope(line):
    # Normalize the line vector
    line = line / np.sqrt(line[0]**2 + line[1]**2)
    # Calculate the slope
    slope = -line[0] / line[1]
    return slope

def get_best_rotation_RANSAC(im,t = 0.01):
    in_liers = 0
    num_liers = 0
    best_line = [0,0,0]
    
    points = np.argwhere(im)

    for i in range(500):
        lNew = make_random_line(points)
        

        new_inliers, new_num_liers = get_inliers(lNew,points,t)
        
        if num_liers < new_num_liers[0]:
            in_liers = new_inliers
            best_line = lNew
            num_liers = new_num_liers[0]
        
    bester_line = pca_line(in_liers)

    deg = np.rad2deg(np.arctan(homogeneous_line_slope(bester_line)))+90
    
    return deg

im = cv.imread("key_images/keyimg17.png")
im = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
im_median = cv.medianBlur(im,5)

thresh = cv.adaptiveThreshold(im_median, 255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 29, 4)




displayImages([im,thresh],["","gray","gray","gray"])
plt.show()