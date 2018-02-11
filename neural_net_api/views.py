from django.http import Http404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import os,cv2,base64
import numpy as np
import pickle
import time
import tensorflow as tf
from neural_net_api import utils
from neural_net_api.NN_models.neuralnet import NeuralNet as net

class FloorDetection(APIView):
	"""
	finds the floor in a picture
	"""
	def get(self, request,format=None):
		n = net('floor_detection')
		data={'image':'hello'}
		input_image = utils.decode_string_image(data.get('image',None))
		return  Response({'name':'Yueng'},status=status.HTTP_405_METHOD_NOT_ALLOWED)

	def post(self,request,format=None):
		data = request.data
		input_image = utils.decode_string_image(data.get('image',None))
		result = net('floor_detection').run_inference_on_image(input_image)
		return  Response({'result':result},status=status.HTTP_200_OK)

class WaterDetection(APIView):
	"""
	finds the water puddle in a picture. This method is used for the original water model with 4 dimensions (RGB + Laplacian edge),
	the original floor detection with 3 dimensions (RGB), the water-floor integration option 1 with 5 dimensions (RGB + Laplacian edge + b/w floor image),
	and the water-floor integration option 2 with 4 dimensions (RGB floor output + Laplacian edge). To change the model to use, we need to execute the 
	"""
	def get(self, request, format=None):
		return Response('respon', status=status.HTTP_200_OK)

	def post(self, request, format=None):
		data = request.data
		result = {'result':rst}
		return Response(result, status=status.HTTP_200_OK)

class ObjectDetection(APIView):
	"""
	finds objects on the floor in a picture
	"""
	def get(self, request,format=None):
		print("GET","Hello ObjectDetection")
		return  Response({'name':'Yueng'},status=status.HTTP_405_METHOD_NOT_ALLOWED)

	def post(self,request,format=None):
		print("POST","Hello ObjectDetection")
		print(request.data)
		return  Response(None,status=status.HTTP_202_ACCEPTED)

