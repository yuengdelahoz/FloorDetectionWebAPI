import zerorpc
import cv2
import pickle
import time
import sys
import numpy as np
import os, os.path
import collections

s = time.time()
c = zerorpc.Client()
# Connect to the Water Detection RPC server, which in this case is listening in address: 127.0.0.1:4242
c.connect("tcp://127.0.0.1:4242")
path = os.path.dirname(__file__)
# In order to calculate the average execution time for the floor detection, the water detection and the overall 
# process, we compute these metrics for each image in the validation dataset and then average the individual results
validationImages = np.load('/home/raulreu/WaterDetectionNN/WaterDetection/Dataset/npyFiles/validationImages.npy')
counter = 0
ftimes = []
wtimes = []
ttimes = []
for valImg in validationImages:
	img1 = cv2.imread('/home/raulreu/WaterDetectionNN/WaterDetection/Dataset/INPUT/'+valImg)
	img_serialized_1 = pickle.dumps(img1,protocol=0)
	retimg,ftime,wtime,ttime = c.run_inference_on_image(img_serialized_1)
	ftimes.append(ftime)
	wtimes.append(wtime)
	ttimes.append(ttime)
	print('It:',counter)
	counter += 1
print(time.time() - s, 'segs')
print('Avg floor time:',np.mean(ftimes))
print('Avg water time:',np.mean(wtimes))
print('Avg total time:',np.mean(ttimes))
