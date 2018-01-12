from darkflow.net.build import TFNet
import cv2

import tensorflow as tf, sys

import http.server
import socketserver
import urllib.request

import json
import numpy as np
from hashlib import md5
import os
import cgi
import uuid

import time
timestamp = lambda: int(round(time.time() * 1000))

CLASSIFY_DIMENSION = 416

SOURCE_MAXLEN = 4 * 1058476
SOURCE_FORMATS = ('image/jpeg', 'image/png', 'image/webp')

IMAGE_DIR_LOCAL = '/var/www/html/images/detection/'
IMAGE_DIR_SERVE = 'http://192.168.56.102/images/detection/'


COLORS = [
    (255,0,0), (128,0,128), (255,0,255),
    (0,255,0), (128,128,0), (255,255,0),
    (0,0,255), (0,128,128), (0,255,255),
    (128,0,0), (0,128,0), (0,0,128),
    (192,192,192), (128,128,128), (255,255,255), (0,0,0),
]

PORT = 8000


def md5digest(raw):
    m = md5()
    m.update(raw)
    return m.hexdigest()


def resize_n_pad( fileobj, dimension ):
    # Load as opencv image object
    # Resize and pad if necessary
    # Return opencv image object for imaging service

    imdata = fileobj.read()
    file_bytes = np.asarray( bytearray( imdata ), dtype=np.uint8 )
    img = cv2.imdecode( file_bytes, 1 )

    height, width, channels = img.shape
    # Resize
    if width >= height: # Landscape
        height = int(dimension * height / width)
        width = dimension
    else: # Portrait
        width = int(dimension * width / height)
        height = dimension
    resized_image = cv2.resize(img, (width, height))
    # Pad
    if width < dimension: # Too wide
        bleft = int((dimension-width)/2)
        bright = dimension - bleft - width
        resized_image = cv2.copyMakeBorder(resized_image, top=0, bottom=0, left=bleft, right=bright, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
    elif height < dimension: # Too tall
        btop = int((dimension-height)/2)
        bbottom = dimension - btop - height
        resized_image = cv2.copyMakeBorder(resized_image, top=btop, bottom=bbottom, left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )

    return resized_image


class IntelligentModels():

    mYolo = None
    mInceptionV3 = None

    OPTIONS = {
        'inceptionv3' : {
            'pre-trained' : {
                'graph' : '/usr/lib/cgi-bin/classify_image/imagenet/classify_image_graph_def.pb',
                'output' : 'softmax:0',
                'label' : '/usr/lib/cgi-bin/classify_image/imagenet/imagenet_labels_sorted.txt',
            },
            'flowers' : {
                'graph' : '/usr/lib/cgi-bin/classify_image/retrained/flowers/retrained_graph.pb',
                'output' : 'final_result:0',
                'label' : '/usr/lib/cgi-bin/classify_image/retrained/flowers/retrained_labels.txt',
            },
        }
    }


    def detect(self, imgcv, service):
        if service=='/detection/yolo':
            return self.yolo(imgcv)
        elif service=='/classify/inceptionv3':
            return self.inceptionv3(imgcv)


    def inceptionv3(self, imgcv):
        if not self.mInceptionV3:
            # Unpersists graph from file
            gpath = self.OPTIONS['inceptionv3']['pre-trained']['graph']
            with tf.gfile.FastGFile(gpath, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(graph_def, name='')

        with tf.Session() as sess:
            # Loads label file, strips off carriage return
            lpath = self.OPTIONS['inceptionv3']['pre-trained']['label']
            label_lines = [line.rstrip() for line 
                            in tf.gfile.GFile(lpath)]

            # Feed the image_data as input to the graph and get first prediction
            tname = self.OPTIONS['inceptionv3']['pre-trained']['output']
            softmax_tensor = sess.graph.get_tensor_by_name(tname)
            
            # Read in the image_data
            #image_data = tf.gfile.FastGFile('/var/www/html/images/detection/ff8254fba9170f90652b87fcf1604a75.jpg', 'rb').read()
            image_data = cv2.imencode('.jpg', imgcv)
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data[1].tostring()})
            
            # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-5:][::-1]
            
            jsonresult = []
            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                jsonresult.append({
                    'label' : human_string,
                    'score' : ' %.5f' % (score)
                })

            return jsonresult


    def yolo(self, imgcv):
        if not self.mYolo:
            #YOLO_CONFIG = './cfg/tiny-yolo-voc.cfg'
            #YOLO_WEIGHTS = './bin/tiny-yolo-voc.weights'
            options = {"model": './cfg/yolo.cfg', "load": './bin/yolo.weights', "threshold": 0.5}
            self.mYolo = TFNet(options)
            print('Yolo detector initialized')
            print()

        result = self.mYolo.return_predict(imgcv)

        height, width, channels = imgcv.shape
        cindex = 0
        jsonresult = []
        for item in result:
            # Draw preview
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

        return jsonresult


IMODELS = IntelligentModels()

class Handler(http.server.SimpleHTTPRequestHandler):

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain;charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.wfile.write(bytes('', "utf8"))


    def do_POST(self):

        print()
        print(self.path)
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json;charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        length = int(self.headers['Content-Length'])
        json_response = {
            'header' : {
                'service' : self.path,
                'date_changed' : '2018-01-12',
                'err_no' : 500,
                'err_msg' : 'Unexpected server error'
            },
            'result' : {
            },
        }
        if length > SOURCE_MAXLEN:
            json_response['header']['err_no'] = 400
            json_response['header']['err_msg'] = 'Image data too large (max '+str(SOURCE_MAXLEN)+' bytes)'
            return

        # Parse multipart data using FieldStorage (so the code can be reused in cgi scripts)
        ctype, pdict = cgi.parse_header(self.headers['Content-Type'])
        if ctype == 'multipart/form-data':
            form = cgi.FieldStorage(
                fp = self.rfile,
                headers = self.headers,
                environ = {'REQUEST_METHOD':'POST'})
            if form.list:
                for item in form.list:
                    if item.file and item.type in SOURCE_FORMATS:
                        # Handling a file
                        now = timestamp()
                        imgcv = resize_n_pad(item.file, 416) # Request with image url
                        results = IMODELS.detect(imgcv, self.path)
                        json_response['result']['annotations'] = results
                        json_response['result']['rid'] = item.name

                        # Save image for preview
                        fname = str(uuid.uuid4())+'.jpg'
                        outpath = os.path.join( IMAGE_DIR_LOCAL, fname )
                        cv2.imwrite(outpath, imgcv)
                        json_response['result']['preview'] = os.path.join( IMAGE_DIR_SERVE, fname )

                        json_response['result']['exec_time'] = timestamp() - now
                        json_response['header']['err_no'] = 0
                        json_response['header']['err_msg'] = 'Success'
        else:
            json_response['header']['err_no'] = 400
            json_response['header']['err_msg'] = 'Only support Content-Type: multipart/form-data; boundary=...'
            
        print(json_response)
        self.wfile.write(bytes(json.dumps(json_response), "utf8"))


socketserver.TCPServer.allow_reuse_address = True
httpd = socketserver.TCPServer(("", PORT), Handler)
print("serving at port", PORT)
try:
    httpd.serve_forever()
except KeyboardInterrupt:
    print("releasing port", PORT)
