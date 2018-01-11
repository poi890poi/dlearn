from darkflow.net.build import TFNet
import cv2

options = {"model": "/df/darkflow/cfg/tiny-yolo-voc.cfg", "load": "/df/darkflow/bin/tiny-yolo-voc.weights", "threshold": 0.5}

tfnet = TFNet(options)

print('TFNet initialized')

imgcv = cv2.imread("/df/darkflow/sample_img/sample_person.jpg")
result = tfnet.return_predict(imgcv)
print(result)