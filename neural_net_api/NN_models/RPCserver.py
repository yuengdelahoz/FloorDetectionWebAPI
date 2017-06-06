import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import cv2
import zerorpc
import pickle

class NeuralNetRPC(object):
	def __init__(self):
		print('Initiating server')
		self.load_graph('model.pb')
		print("Graph loaded.","Server ready")


	def load_graph(self,frozen_graph_filename):
		# We load the protobuf file from the disk and parse it to retrieve the 
		# unserialized graph_def
		with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())

		# Then, we can use again a convenient built-in function to import a graph_def into the 
		# current default Graph
		with tf.Graph().as_default() as self.g:
			tf.import_graph_def(
				graph_def, 
				input_map=None, 
				return_elements=None, 
				name="prefix", 
				op_dict=None, 
				producer_op_list=None
			)
		return self.g

	def paintOrig(self,supimgVector,img):
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

	def run_inference_on_image(self,img_bytes):
		img = np.asarray(bytearray(img_bytes), dtype="uint8")
		input_image = cv2.imdecode(img, cv2.IMREAD_COLOR)
		print('running inference on image')
		image = np.array(input_image,ndmin=4)
		# for op in g.get_operations():
			# print(op.name)
		x = self.g.get_tensor_by_name("prefix/input_images:0")
		keep_prob = self.g.get_tensor_by_name("prefix/keep_prob:0")
		output= self.g.get_tensor_by_name("prefix/superpixels:0")
		with tf.Session(graph=self.g) as sess:
			result = sess.run(output,feed_dict={x:image,keep_prob:1.0})
			print (result.shape)
			paintedImg = self.paintOrig(result.ravel(),input_image)
			cv2.imwrite('out_image.jpg',paintedImg)
		return "Done"

	def run_string(self, string):
		print(string)

	def save_image(self,img_bytes):
		image = np.asarray(bytearray(img_bytes), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)
		cv2.imwrite('rr_img.jpg',image)
		 

if __name__ == "__main__":
	server = NeuralNetRPC()
	s = zerorpc.Server(server)
	s.bind("tcp://127.0.0.1:4242")
	s.run()
