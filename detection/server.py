from darkflow.net.build import TFNet
import cv2

import http.server
import socketserver
import urllib.request

import json
import numpy as np
from hashlib import md5
import os

import time
timestamp = lambda: int(round(time.time() * 1000))

CLASSIFY_MAXLEN = 1058476

IMAGE_DIR = '/var/www/html/images/detection/'
IMAGE_DIR_MAPPED = 'http://192.168.56.102/images/detection/'

#YOLO_CONFIG = './cfg/tiny-yolo-voc.cfg'
#YOLO_WEIGHTS = './bin/tiny-yolo-voc.weights'
YOLO_CONFIG = './cfg/yolo.cfg'
YOLO_WEIGHTS = './bin/yolo.weights'
YOLO_THRESHOLD = 0.5

COLORS = [
    (255,0,0), (128,0,128), (255,0,255),
    (0,255,0), (128,128,0), (255,255,0),
    (0,0,255), (0,128,128), (0,255,255),
    (128,0,0), (0,128,0), (0,0,128),
    (192,192,192), (128,128,128), (255,255,255), (0,0,0),
]

PORT = 8000

options = {"model": YOLO_CONFIG, "load": YOLO_WEIGHTS, "threshold": 0.5}

tfnet = TFNet(options)

print('TFNet initialized')
print()


def md5digest(raw):
    m = md5()
    m.update(raw)
    return m.hexdigest()


def run_inference_on_image(image_data, outpath = ''):
    file_bytes = np.asarray( bytearray( image_data ), dtype=np.uint8 )
    imgcv = cv2.imdecode( file_bytes, 1 )
    result = tfnet.return_predict(imgcv)

    height, width, channels = imgcv.shape
    cindex = 0
    jsonresult = []
    for item in result:
        cv2.putText(imgcv, item['label'], (item['topleft']['x']+3,item['bottomright']['y']-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[cindex], 1) 
        cv2.rectangle(imgcv, (item['topleft']['x'],item['topleft']['y']), (item['bottomright']['x'],item['bottomright']['y']), COLORS[cindex], 2)
        jsonresult.append({
            'label': item['label'],
            'confidence': str(item['confidence']),
            'topleft': (item['topleft']['x'], item['topleft']['y']),
            'bottomright': (item['bottomright']['x'], item['bottomright']['y']),
        })
        cindex += 1
        if cindex > 15: cindex = 0

    if outpath:
        cv2.imwrite(outpath, imgcv)

    return jsonresult


class Handler(http.server.SimpleHTTPRequestHandler):

    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type',
                         'application/json;charset=utf-8')
        self.end_headers()

        length = int(self.headers['Content-Length'])
        dtype = self.headers['Arobot-Data-Type']
        rid = self.headers['Arobot-Request-Id']
        json_response = {
            'header' : {
                'api' : 'detection',
                'date_changed' : '2018-01-10',
                'request_id' : rid,
                'request_data_type' : dtype,
                'err_no' : 500,
                'err_msg' : 'Unexpected server error'
            },
            'detection' : {
                'model' : 'tiny-yolo-voc',
            },
        }
        api = json_response['header']['api']
        if length > CLASSIFY_MAXLEN:
          json_response['header']['err_no'] = 400
          json_response['header']['err_msg'] = 'Image data too large (max '+str(CLASSIFY_MAXLEN)+' bytes)'
          return

        if dtype=='url':
          url = self.rfile.read(length).decode('utf-8')
          with urllib.request.urlopen(url) as img:
            img_data = img.read()
            #image_data = tf.gfile.FastGFile(url, 'rb').read()
            json_response[api]['url'] = url
            now = timestamp()
            fname = md5digest(url.encode('utf-8')) + '.jpg'
            outpath = os.path.join( IMAGE_DIR, fname )
            json_response[api]['objects'] = run_inference_on_image(img_data, outpath)
            json_response[api]['url'] = os.path.join( IMAGE_DIR_MAPPED, fname )
            json_response[api]['exec_time'] = timestamp() - now
            json_response['header']['err_no'] = 0
            json_response['header']['err_msg'] = 'Success'
        elif dtype=='bin':
          img_data = self.rfile.read(length)
          fname = md5digest(img_data) + '.jpg'
          outpath = os.path.join( IMAGE_DIR, fname )
          json_response[api]['objects'] = run_inference_on_image(img_data, outpath)
          json_response[api]['url'] = os.path.join( IMAGE_DIR_MAPPED, fname )
          json_response['header']['err_no'] = 0
          json_response['header']['err_msg'] = 'Success'
          pass
        else:
          json_response['header']['err_no'] = 400
          json_response['header']['err_msg'] = 'Unsupported data type'
          pass

        print(json_response)
        self.wfile.write(bytes(json.dumps(json_response), "utf8"))


socketserver.TCPServer.allow_reuse_address = True
httpd = socketserver.TCPServer(("", PORT), Handler)
print("serving at port", PORT)
try:
    httpd.serve_forever()
except KeyboardInterrupt:
    print("releasing port", PORT)
