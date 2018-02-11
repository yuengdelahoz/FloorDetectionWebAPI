from django.http import Http404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import os,cv2,base64
import numpy as np
import pickle
import time
import tensorflow as tf
import traceback,sys
from neural_net_api import utils
from neural_net_api.NN_models.neuralnet import NeuralNet as net

class FallPrevention(APIView):
	def post(self,request,format=None):
		data = request.data
		try:
			b64_image = data.get('image',None)
			model = data.get('model',None)
			print(model)
			input_image = utils.decode_string_image(b64_image)
			result = net(model).run_inference_on_image(input_image)
			return  Response({'result':result},status=status.HTTP_200_OK)
		except:
			# traceback.print_exc(file=sys.stdout)
			return  Response({'result':[[1,2,3,4,5]]},status=status.HTTP_200_OK)

