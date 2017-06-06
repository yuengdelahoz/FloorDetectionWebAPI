from django.http import Http404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from neural_net_api.NN_models import test
import os,cv2,base64
import numpy as np
import zerorpc
import pickle
import time


class FloorDetection(APIView):
	"""
	finds the floor in a picture
	"""
	def get(self, request,format=None):
		print (request.data)
		return Response("Hello",status=status.HTTP_200_OK)

	def post(self,request,format=None):
		data = request.data
		image_name = data['name']
		image_str = data['image']
		img_bytes = base64.b64decode(image_str)


		s = time.time()
		c = zerorpc.Client()
		c.connect("tcp://127.0.0.1:4242")
		rst = c.run_inference_on_image(img_bytes)

		# c.save_image(img_bytes)
		print(time.time() - s, 'segs')

		# rst = c.run_inference_on_image(img_serialized)

		return  Response(rst,status=status.HTTP_200_OK)

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

class TokenRegistration(APIView):
	"""
		Registers firebase tokens on the sql database
	"""
	def get(self, request,format=None):
		return  Response({'name':'Yueng'},status=status.HTTP_405_METHOD_NOT_ALLOWED)

	def post(self,request,format=None):
		return  Response(None,status=status.HTTP_202_ACCEPTED)


