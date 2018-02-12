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

class NeuralNet(object):
	# Model paths
	dir_path = os.path.dirname(os.path.realpath(__file__))
	model_path = dict()
	model_path.update({'floor_detection':dir_path+'/frozen_models/floor_detection.pb'})
	model_path.update({'water_detection':dir_path+'/frozen_models/water_detection.pb'})
	model_path.update({'object_detection':dir_path+'/frozen_models/object_detection.pb'})
	model_path.update({'distance_detection':dir_path+'/frozen_models/distance_detection.pb'})

	# Model graphs
	model_graph = dict()
	def __init__(self,model):
		# print('Starting Neural Net')
		self.graph = self.load_graph(model)

	def load_graph(self,model):
		# We load the protobuf file from the disk and parse it to retrieve the 
		# unserialized graph_def
		graph = NeuralNet.model_graph.get(model,None)
		if graph is not None:
			# print("Loading graph from memory")
			return graph

		try:
			graphdef_path = NeuralNet.model_path.get(model,None)
			# print("Loading graph from disk")
			with tf.gfile.GFile(graphdef_path, "rb") as f:
				graph_def = tf.GraphDef()
				graph_def.ParseFromString(f.read())

			# Load and return graph
			with tf.Graph().as_default() as g:
				tf.import_graph_def(
					graph_def, 
					name ='fallprevention'
				)
				NeuralNet.model_graph.update({model:g})
				return g
		except:
			print('Graph could not be loaded from disk')

	def run_inference_on_image(self,input_image):
		start_time = time.time()
		image = (input_image-128)/128
		image = np.array(image,ndmin=4)

		with tf.Session(graph=self.graph) as sess:
			xf = self.graph.get_tensor_by_name("fallprevention/input_images:0")
			keep_probf = self.graph.get_tensor_by_name("fallprevention/keep_prob:0")
			outputf = self.graph.get_tensor_by_name("fallprevention/superpixels:0")
			result = sess.run(outputf,feed_dict={xf:image,keep_probf:1.0})
			end_time = time.time()
		return result,int(round((end_time-start_time)*1000))


