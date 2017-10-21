from django.http import Http404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import os,cv2,base64
import numpy as np
import zerorpc
import pickle
import time

class WaterDetection(APIView):
	"""
	finds the water puddle in a picture. This method is used for the original water model with 4 dimensions (RGB + Laplacian edge),
	the original floor detection with 3 dimensions (RGB), the water-floor integration option 1 with 5 dimensions (RGB + Laplacian edge + b/w floor image),
	and the water-floor integration option 2 with 4 dimensions (RGB floor output + Laplacian edge). To change the model to use, we need to execute the 
	RPC server we want, which will run in localhost port 4242
	"""
	def get(self, request, format=None):
		print(request.data)
		return Response("Hello", status=status.HTTP_200_OK)

	def post(self, request, format=None):
		data = request.data
		image_str = data['image']
		#s = time.time()
		c = zerorpc.Client()
		c.connect("tcp://127.0.0.1:4242")
		rst = c.run_inference_on_image(image_str)
		#print(time.time() - s, "segs")
		return Response(rst, status=status.HTTP_200_OK)

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


