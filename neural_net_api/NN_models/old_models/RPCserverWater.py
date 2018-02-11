import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import cv2
import base64
import zerorpc
import pickle
from wand.image import Image
from random import shuffle
import sys
import time
from script import *

"""
	This RPC server loads the original water detection model whose input has 4 dimensions (original 500X500 RGB image + Laplacian edge detection). The server will compute
	the gradient image and it only receives the 500x500 RGB image encoded as base64 string
"""
class NeuralNetRPC(object):
	def __init__(self):
		print('Initiating server')
		# We load the water detection model from the frozen model file
		self.w_pref = "prefix_water"
		self.w = self.load_graph('v2_water_model_original.pb',self.w_pref)
		print("Graph loaded.","Server ready")

	def load_graph(self,frozen_graph_filename,model_prefix):
		# We load the protobuf file from the disk and parse it to retrieve the 
		# unserialized graph_def
		with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())

		# Then, we can use again a convenient built-in function to import a graph_def into the 
		# current default Graph
		with tf.Graph().as_default() as g:
			tf.import_graph_def(
				graph_def, 
				input_map=None, 
				return_elements=None, 
				name=model_prefix, 
				op_dict=None, 
				producer_op_list=None
			)
			targetGraph = g
		return targetGraph

	def run_inference_on_image(self,img_bytes):
		"""Runs inference on the RGB 500x500 input image. First, the water detection model is loaded."""
		# tini = time.time()
		# Use CPU instead of GPU so that we can load multiple models and the GPU can be used for other purposes
		session_conf = tf.ConfigProto(device_count={'CPU' : 1, 'GPU' : 0},
                allow_soft_placement=True,
                log_device_placement=False)

		btes = base64.b64decode(img_bytes)
		input_image = np.frombuffer(btes, np.uint8)
		input_image = cv2.imdecode(input_image, cv2.IMREAD_COLOR)
		img = np.array(input_image,ndmin=3)

		# Compute the edge gradient image
		ein = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		edgeimg = cv2.Laplacian(ein, cv2.CV_8U, ksize = 3)
		height,width,channels = img.shape

		waterInputImg = np.empty((width,height,4),dtype=np.uint8)
		waterInputImg[:,:,0:3] = img
		waterInputImg[:,:,3] = edgeimg
		# Run inference using the water detection model
		waterInputImg = waterInputImg/255.0;
		xw = self.w.get_tensor_by_name(self.w_pref+"/input_images:0")
		keep_probw = self.w.get_tensor_by_name(self.w_pref+"/keep_prob:0")
		outputw = self.w.get_tensor_by_name(self.w_pref+"/superpixels:0")
		with tf.Session(graph=self.w, config=session_conf) as sess:
			# waterini = time.time()
			result = sess.run(outputw,feed_dict={xw:np.array(waterInputImg,ndmin=4),keep_probw:1.0})
			# waterend = time.time()
			painted = paintOrig(result.ravel(),img)
			encoded = cv2.imencode(".jpg",painted)[1]
			str_image = base64.b64encode(encoded)
		# tend = time.time()
		# watertime = waterend-waterini
		# totaltime = tend-tini
		# print('Water model inference:',watertime,'segs')
		# print('Total inference:',totaltime,'segs')
		return str_image

if __name__ == "__main__":
	server = NeuralNetRPC()
	s = zerorpc.Server(server)
	s.bind("tcp://127.0.0.1:4242")
	s.run()

