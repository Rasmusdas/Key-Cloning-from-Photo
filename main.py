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
import glob
from skimage.morphology import (erosion, dilation)
import os
from skimage.morphology import disk 
from skimage import measure
from skimage.color import label2rgb

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
                
def load_image(imgName,scaling):
    img = cv.imread(imgName)
    img = cv.resize(img,(img.shape[1]//scaling,img.shape[0]//scaling))
    return img

def edge_detection(img,its,kernelRadius = 4,medianKernelRadius = 7,iterationKernelRadius = 7):
    
    # Blurring the image
    img = cv.medianBlur(img,medianKernelRadius)

    # Edge detection with simple kernel
    kernel = np.array([[-1 for j in range(kernelRadius*2+1)] for i in range(kernelRadius*2+1)])
    
    kernel[kernelRadius,kernelRadius] = (kernelRadius*2+1)**2-1
    keyEdge = cv.filter2D(img,-1,kernel)
    
    # Noise Removal Iterations
    tH = 240
    for i in range(its):
        keyEdge[keyEdge < tH] = 0 
        keyEdge = cv.GaussianBlur(keyEdge,(iterationKernelRadius,iterationKernelRadius),1)
    
    # End noise removal
    keyEdge[keyEdge < tH] = 0 


    return keyEdge

def amplify_noise(img,size):
    dist_kernel = np.ones((size,size))
    return cv.filter2D(img,-1,dist_kernel)

def fill_key(img):
    img = np.copy(img)
    
    img = img.T

    img[img > 0] = 255
    
    for i in range(img.shape[0]):
        min_val = np.argmax(img[i])
        max_val = np.argmax(np.flip(img[i]))
        
        if min_val == 0: continue
        if abs(min_val-len(img[i])+max_val) < 35:
            if np.argmax(img[i-1]) - len(img[i]) + np.argmax(np.flip(img[i-1])) < 35:
                continue
            min_val = np.argmax(img[i-1])
            max_val = np.argmax(np.flip(img[i-1]))
        img[i,min_val:len(img[i])-max_val] = 255
        
    img = img.T
    return img

def remove_blobs(img,stats):
    avg = np.average(stats[1:,4])
    
    stats = stats[stats[:,4] <= (avg*8)]
    
    fixed_img = np.copy(img)
    
    for stat in stats:
        fixed_img[stat[1]:stat[1]+stat[3],stat[0]:stat[0]+stat[2]] = 0
        
    return fixed_img

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

def visualize_blobs(img,labels,ids,stats,out_dir,out_name):
    avg = np.average(stats[1:,4])

    imgCopy = np.copy(img)
    
    imgCopy = cv.cvtColor(imgCopy,cv.COLOR_GRAY2RGB)

    for i in range(1,labels):
        area = stats[i,4]
        if area < avg*8:
            mask = ids == (i)
            imgCopy[mask] = (255,0,0)
            if i % 10 == 0:
                print(f"Visualizing Blobs: {i}/{labels}")
        

    
    plt.imshow(imgCopy)
    plt.show()

    plt.imsave(out_dir+out_name+"marked_blobs.png",imgCopy)
    
def visualize_hough_lines(img,threshold,out_dir,out_name,overlayImage = None):
    if overlayImage is not None:
        imgCopy = np.copy(overlayImage)
    else:
        imgCopy = np.copy(img)
    imgCopy = cv.cvtColor(imgCopy,cv.COLOR_GRAY2RGB)

    lines = cv.HoughLines(img, 1, math.pi/180, threshold)

    lines = sorted(lines, key=lambda x: x[0][0], reverse=True)

    line = lines[:10]
    for li in line:
            rho = li[0][0]
            theta = li[0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(imgCopy, pt1, pt2, (255,0,0), 3, cv.LINE_AA)

        

    plt.imshow(imgCopy)
    plt.show()

    plt.imsave(out_dir+out_name+"hough_lines.png",imgCopy)

def erode_key(filled_key,erosionFootprint):
    eroded = filled_key
    if erosionFootprint > 0:
        print("Eroding Key")
        footprint = disk(erosionFootprint)
        eroded = erosion(filled_key, footprint)
        
    if erosionFootprint < 0:
        print("Dilating Key")
        footprint = disk(-erosionFootprint)
        eroded = dilation(filled_key, footprint)
        
    return eroded
    


def generate_key(imgName,downsizeFactor = 1,kernelSize = 7, noiseReductionIts = 1,medianBlurSize = 9, noiseRemoval = 2,lineDetectingThreshold=1000, erosionFootprint = -1, keyHeadSize = 30.18,outName="",outDir = "out/"):
    np.set_printoptions(suppress=True) 
    print("Starting Key Info Extraction")
    img = load_image(imgName,downsizeFactor)
    
    img_gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)

    print("Detecting Key")
    keyEdge = edge_detection(img_gray,noiseReductionIts,kernelSize,medianBlurSize)
    
    print("Removing Noise")
    labels, ids, stats, _ = cv.connectedComponentsWithStats(amplify_noise(keyEdge,noiseRemoval),8,cv.CV_32S)    

    blobs_removed = remove_blobs(keyEdge,stats)
    
    #visualize_blobs(keyEdge,labels,ids,stats,out_dir,out_name)
    
    print("Rotating Image")
    rho, degree = get_best_rotation(blobs_removed,lineDetectingThreshold)
    
    #visualize_hough_lines(fixed_small_rem,lineDetectingThreshold,out_dir,out_name)
    
    fixed_rotation = rotate(blobs_removed,-degree)
    
    print("Fixing Key")
    filled_key = fill_key(fixed_rotation)

    eroded = erode_key(filled_key,erosionFootprint)
    
    print("Detecting Bottom Of Key")
    edge_test = edge_detection(eroded,1,kernelSize,1,1)
    
    print("Removing Key Head")
    #visualize_hough_lines(edge_test,lineDetectingThreshold,out_dir,out_name,filled_key)
    
    lines= cv.HoughLines(edge_test, 0.5, math.pi/180.0, int(lineDetectingThreshold))
    new_rho = lines[0][0][0]+3 
    
    ind = np.asarray(eroded >=250).nonzero()
    minx = ind[0][0]
    maxx = ind[0][-1]
    
    eroded_bounded = eroded[minx:int(new_rho),np.min(ind[1]):np.max(ind[1])]
    
    bounded_rot = rotate(img,-degree)[minx:int(new_rho),np.min(ind[1]):np.max(ind[1])]
    
    print("Creating Overlay")
    
    overlay = np.copy(bounded_rot)
    
    overlay[eroded_bounded != 0] += (np.uint8(255),np.uint8(0),np.uint8(0))
    overlay[eroded_bounded != 0] *= (np.uint8(255),np.uint8(0),np.uint8(0))
    
    print("Saving Keys")
    cv.imwrite(outDir+outName+"out.bmp",255-eroded_bounded)
    plt.imsave(outDir+outName+"outBounded.png",cv.cvtColor(eroded_bounded,cv.COLOR_GRAY2BGR))
    plt.imsave(outDir+outName+"orgbound.png",bounded_rot)
    plt.imsave(outDir+outName+"filled_key.png",cv.cvtColor(filled_key,cv.COLOR_GRAY2BGR))
    cv.imwrite(outDir+outName+"outFull.bmp",255-eroded)
    plt.imsave(outDir+outName+"orgrot.png",rotate(img,-degree))
    plt.imsave(outDir+outName+"overlay.png",overlay)
    plt.imsave(outDir+outName+"rotatedFix.png",cv.cvtColor(fixed_rotation,cv.COLOR_GRAY2BGR))
    plt.imsave(outDir+outName+"org.png",img)
    plt.imsave(outDir+outName+"keyEdge.png",cv.cvtColor(keyEdge,cv.COLOR_GRAY2BGR))
    plt.imsave(outDir+outName+"smallRemoved.png",cv.cvtColor(blobs_removed,cv.COLOR_GRAY2BGR))
    os.system(f'potrace {outDir}out.bmp -b svg -o {outDir}generated_out.svg')
    
    # Display images
    print("Displaying images")
    imgs = [img,img_gray,keyEdge,amplify_noise(keyEdge,noiseRemoval),blobs_removed,fixed_rotation, filled_key,eroded,edge_test,eroded_bounded,overlay]
    displayImages(imgs,["","gray","gray","gray","gray","gray","gray","gray","gray","gray","gray","gray","gray"])

    plt.show()

    print("Finished with key image")

def generate_keys_multi(in_dir,maxWorkers = 8):
    pics = glob.glob(in_dir)
    working = []
    for i,pic in enumerate(pics):
        p = Process(target=generate_key_process, args=(str(pic)[11:],str(pic)))
        working.append(p)
        p.start()
        
        if(len(working) == maxWorkers):
            for proc in working:
                proc.join()
            working = []

def generate_key_process(i,name, downsizeFactor = 2,kernelSize = 8, noiseReductionIts = 1,medianBlurSize = 9, noiseRemoval = 2,lineDetectingThreshold=1000, erosionFootprint = -1, keyHeadSize = 30.18,outName="",outDir = "out/"):
    print("\nUsing settings:")
    print(f"imgName: {name}")
    print(f"downsizeFactor: {downsizeFactor}")
    print(f"kernelSize: {kernelSize}")
    print(f"noiseReductionIts: {noiseReductionIts}")
    print(f"medianBlurSize: {medianBlurSize}")
    print(f"noiseRemoval: {noiseRemoval}")
    print(f"lineDetectingThreshold: {lineDetectingThreshold}")
    print(f"erosionFootprint: {erosionFootprint}")
    print(f"keyHeadSize: {keyHeadSize}")
    print(f"outName: {outName}")
    print(f"outDir: {outDir}")
    
    
    thresh = lineDetectingThreshold
    finished = False
    attempt = 0
    while not finished:
        try:
            return generate_key(name,downsizeFactor,kernelSize,noiseReductionIts,medianBlurSize,noiseRemoval,thresh,erosionFootprint,keyHeadSize,i+outName,outDir)
        except Exception as error:
            thresh = int(thresh*0.85)
            print(error)
            attempt+=1
            print("Attempt " + str(attempt))
            if attempt == 5:
                print("Image: " + name + " failed")
                return

from multiprocessing import Process
if __name__ == "__main__":
    print(cv.__version__)
    #generate_keys_multi("key_images/*.png",8)
    generate_key_process("","key_images/keyimg26.png")
