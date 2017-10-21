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
from script import paintOrig

"""
	This RPC server loads the original floor detection model and performs the inference on the input 500x500 RGB image
"""
class NeuralNetRPC(object):
	def __init__(self):
		print('Initiating server')
		# We load the floor detection model from the frozen model file
		self.f_pref = "prefix_floor"
		self.f = self.load_graph('floor_model.pb',self.f_pref)
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
		"""Runs inference on the RGB 500x500 input image. First, both the floor and water detection models are loaded.
		We time the execution time of the floor detection model, the water detection model and the whole process for evaluation purposes"""
		# tini = time.time()
		# Use CPU instead of GPU so that we can load multiple models and the GPU can be used for other purposes
		session_conf = tf.ConfigProto(device_count={'CPU' : 1, 'GPU' : 0},
                allow_soft_placement=True,
                log_device_placement=False)

		btes = base64.b64decode(img_bytes)
		input_image = np.frombuffer(btes, np.uint8)
		input_image = cv2.imdecode(input_image, cv2.IMREAD_COLOR)
		image = np.array(input_image/255,ndmin=4)

		# Run inference on the image using the floor detection model
		xf = self.f.get_tensor_by_name(self.f_pref + "/input_images:0")
		keep_probf = self.f.get_tensor_by_name(self.f_pref+"/keep_prob:0")
		outputf = self.f.get_tensor_by_name(self.f_pref+"/superpixels:0")
		with tf.Session(graph=self.f, config=session_conf) as sess:
			# floorini = time.time()
			result = sess.run(outputf,feed_dict={xf:image,keep_probf:1.0})
			# floorend = time.time()
			paintedImg = paintOrig(result.ravel(),input_image)
			encoded = cv2.imencode(".jpg",paintedImg)[1]
			str_image = base64.b64encode(encoded)
		# tend = time.time()
		# floortime = floorend-floorini
		# totaltime = tend-tini
		# print('Floor model inference:',floortime,'segs')
		# print('Total inference:',totaltime,'segs')
		return str_image

if __name__ == "__main__":
	server = NeuralNetRPC()
	s = zerorpc.Server(server)
	s.bind("tcp://127.0.0.1:4242")
	s.run()

