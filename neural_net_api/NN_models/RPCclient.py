import zerorpc
import cv2
import pickle
import time
import sys

s = time.time()
c = zerorpc.Client()
port = sys.argv[1]
c.connect("tcp://127.0.0.1:"+port)
img1 = cv2.imread('image.jpg')
img_serialized_1 = pickle.dumps(img1,protocol=0)
print(c.run_inference_on_image(img_serialized_1))
# c.run_string("ITO MANUEL")

# print(c.run_inference_on_image(img_serialized_2))
print(time.time() - s, 'segs')
# cv2.imwrite('out_image.jpg',out_image)
