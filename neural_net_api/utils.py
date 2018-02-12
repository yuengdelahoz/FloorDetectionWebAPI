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

def clear_folder(name):
	if os.path.isdir(name):
		try:
			shutil.rmtree(name)
		except:
			 pass
			# print(name,'could not be deleted')
			# traceback.print_exc(file=sys.stdout)

def create_folder(name,clear_if_exists = True):
	if clear_if_exists:
		clear_folder(name)
	try:
		os.makedirs(name)
		return name
	except:
		return name

def generate_new_input_using_floor_detection(image,superlabel):
	paintedImg = image.copy()
	pos = 0
	for sv in range(0,240,8): # 12 superpixels in the height direction
		for sh in range(0,240,8): # 12 superpixels in the width direction
			if superlabel[pos]==0:
				paintedImg[sv:sv+8,sh:sh+8] = 255
			pos +=1
	return paintedImg

def decode_string_image(b64_str):
	try:
		image_str = base64.b64decode(b64_str)
		np_image = np.frombuffer(image_str, np.uint8)
		mat_image =cv2.imdecode(np_image,cv2.IMREAD_COLOR)
		return mat_image
	except:
		None

def process_result(result, model, input_image):
	if model in 'floor_detection':
		return generate_new_input_using_floor_detection(input_image,result[0])
	else:
		return input_image
			

def log_data(filename,line):
	path = create_folder('execution_times/', False)
	with open(path+filename,'a') as f:
		f.write(line+'\n')


