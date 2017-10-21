import os
import cv2
import numpy as np
from wand.image import Image
from random import shuffle
import sys
import time
import zerorpc
import pickle
import base64

def clearFolder(path):
    print('Clearing',path)
    for jpg_file in os.scandir(path):
        if jpg_file.name.endswith('.jpg')!=-1:
            os.remove(jpg_file.path)

# It reads the input images and detects edges in the images using the Canny edge detector with a minValue threshold of 40 and a maxValue threshold of 60
def detectEdgesCanny():
    clearFolder('EDGES')
    for file in os.scandir('INPUT/'):
        if file.name.endswith('.jpg'):
            img = cv2.imread(file.path)
            edgeimg = cv2.Canny(img, 40, 60)
            cv2.imwrite('EDGES/'+file.name,edgeimg)
            print('Edge detection finished for',file.name)

# It reads the input images and detects edges in the images using the Laplacian edge detector with a kernel size of 3
def detectEdgesLaplacian():
    clearFolder('EDGES')
    for file in os.scandir('INPUT/'):
        if file.name.endswith('.jpg'):
            img = cv2.imread(file.path)
            edgeimg = cv2.Laplacian(img, cv2.CV_64F, ksize = 3)
            cv2.imwrite('EDGES/'+file.name,edgeimg)
            print('Edge detection finished for',file.name)


def resizeImages():
    for file in os.scandir('INPUT/'):
        if file.name.endswith('.jpg'):
            print(file.name)
            img = cv2.imread(file.path)
            lbl = cv2.imread("LABEL/" + file.name,0)
            height, width = img.shape[:2]
            imgres = cv2.resize(img,(500, 500), interpolation = cv2.INTER_AREA)
            lblres = cv2.resize(lbl,(500, 500), interpolation = cv2.INTER_AREA)
            cv2.imwrite('INPUT/'+file.name,imgres)
            cv2.imwrite('LABEL/'+file.name,lblres)
    print('Done resizing')

def cropImages():
    clearFolder('INPUT')
    clearFolder('LABEL')
    i = 0;
    for folder in os.scandir('Originals/'):
        for subfolder in os.scandir(folder.path):
            if subfolder.name == 'originals':
                for file in os.scandir(subfolder.path):
                    if file.name.endswith('.png') or file.name.endswith('.jpg'):
                        print(file.name,'iter',i)
                        img = cv2.imread(file.path)
                        height, width,_ = img.shape
                        lbl = cv2.imread(folder.path + "/labels/" + file.name,0)
                        if height == 1080:
                            cv2.imwrite('INPUT/image-'+str(i)+'.jpg',img)
                            cv2.imwrite('LABEL/image-'+str(i)+'.jpg',lbl)
                            i +=1
                        else:
                            for j in range(0,841,105): # 9 crops
                                imgres = img[j:1080+j,:]
                                lblres = lbl[j:1080+j,:]
                                cv2.imwrite('INPUT/image-'+str(i)+'.jpg',imgres)
                                cv2.imwrite('LABEL/image-'+str(i)+'.jpg',lblres)
                                i +=1
    print('Done cropping')

def createSuperImage(img):
    superImage = list() # empty list where to append Superpixels
    """
    There are 500x500 = 250000 pixels, so every superpixels (1250 in total) has 250000/1250=200 pixels
    The resolution of each superpixel is 20x10 pixels
    img[rows,cols]
    img[0,0] = 0 (black)
    img[0,0] = 255 (white)
    """
    WHITE = 255 * 200 # White value * number of pixels in a superpixel
    sh = 0 # horizontal shift
    sv = 0 # vertical shift
    # iterate over the image and create the 1250 superpixels (25x50)
    for sv in range(0,500,10): # 50 superpixels in the height direction
        for sh in range(0,500,20): # 25 superpixels in the width direction
            rst = np.sum(img[sv:sv+10,sh:sh+20])# sum all the pixel values in the superpixel. img[rows,cols]
            if rst > 0.95 * WHITE : # if superpixel is more than 95% white.
                superImage.append(0)
            else: # if image is 5% black or more
                superImage.append(1)
    return superImage

def createSuperImages():
    iter = 0
    for img in os.scandir('LABEL'):
        if img.name.endswith('.jpg'):
            print(img.name, 'iter', iter)
            name,ext = img.name.split('.')
            IMG = cv2.imread(img.path)
            SUPIMG = createSuperImage(IMG)
            np.save('LABEL-SUP/'+name,np.array(SUPIMG))
            iter +=1

def paintOrig(supimgVector,img):
    """Iterate over original image (color) and paint (red blend) the superpixels that were identified as being part of the floor by the neural network"""
    origImg = img.copy()
    sh = 0 # horizontal shift
    sv = 0 # vertical shift
    height,width = origImg.shape[0:2]
    supix = 0
    for sv in range(0,500,10): # 50 superpixels in the height direction
        for sh in range(0,500,20): # 25 superpixels in the width direction
            if supimgVector[supix]>0.5:
                red =np.zeros(origImg[sv:sv+10,sh:sh+20].shape)
                red[:,:,2] = np.ones(red.shape[0:2])*255
                origImg[sv:sv+10,sh:sh+20] = origImg[sv:sv+10,sh:sh+20]*0.5 + 0.5*red # 90% origin image, 10% red
            supix = supix + 1
    return origImg

def paintImages():
    i = 0
    for img in os.scandir('INPUT'):
        if img.name.endswith('.jpg'):
            name,ext = img.name.split('.')
            origIMG = cv2.imread(img.path)
            SUPIMG = np.load('LABEL-SUP/'+name+".npy")
            pimg = paintOrig(SUPIMG,origIMG)
            cv2.imwrite('PAINTED-IMAGES/'+img.name,pimg)
            print(img.name, 'iter', i)
            i +=1

def createFloorDetectionImages():
	clearFolder('FLOOR')
	i = 0
	client = zerorpc.Client()
	client.connect('tcp://127.0.0.1:4242')
	for img in os.scandir('INPUT'):
		if img.name.endswith('.jpg'):
			origImg = cv2.imread(img.path)
			origImgSerialized = pickle.dumps(origImg,protocol=0)
			res = client.run_inference_on_image(origImgSerialized)
			resImg = base64.b64decode(res)
			with open('FLOOR/'+img.name, 'wb') as file:
				file.write(resImg)
			print(img.name, 'iter',i)
			i += 1
	print('Done')
