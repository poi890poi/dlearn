from darkflow.net.build import TFNet
import cv2

import tensorflow as tf, sys

import face_recognition

from abc import ABC, abstractmethod
 
from socketserver import ThreadingMixIn
from http.server import SimpleHTTPRequestHandler, HTTPServer
import urllib.request
import threading
import urllib.parse

import json
import numpy as np
from hashlib import md5
import os
import cgi
import uuid
from base64 import b64decode

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
CV2_FONTFACE = cv2.FONT_HERSHEY_PLAIN
CV2_FONTSCALE = 1
CV2_THICKNESS = 1
CV2_PADDING = 3

PORT = 8000


def md5digest(raw):
    m = md5()
    m.update(raw)
    return m.hexdigest()


def imgFit( img, dimension ):

    # Load as opencv image object
    # Resize and pad if necessary
    # Return opencv image object for imaging service

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


def resize_n_pad( fileobj, dimension ):

    imdata = fileobj.read()
    file_bytes = np.asarray( bytearray( imdata ), dtype=np.uint8 )
    img = cv2.imdecode( file_bytes, 1 )

    return imgFit( img, dimension )


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
    

class FaceRecognition(metaclass=Singleton):

    DIR_KNOWN_FACES = os.path.join(os.environ['MODEL_DIR'], 'face_recognition/known')
    mFaceNames = []
    mFaceEncodings = []

    def __init__(self):
        print(self.DIR_KNOWN_FACES)
        directory = os.fsencode(self.DIR_KNOWN_FACES)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".jpg"):
                facename = os.path.splitext(filename)[0]
                encpath = os.path.join(self.DIR_KNOWN_FACES, facename + '.npy')

                imgpath = os.path.join(self.DIR_KNOWN_FACES, filename)
                image = face_recognition.load_image_file(imgpath)
                face_encodings = face_recognition.api.face_encodings(image, num_jitters=1)

                if len(face_encodings) > 1:
                    print('Image of known faces should NOT contain more than one face: ' + facename)
                self.mFaceNames.append(facename)
                self.mFaceEncodings.append(face_encodings[0])
                np.save(encpath, face_encodings[0])
                continue
            else:
                continue
        print('Face recognition initialized')
        print()

    def run(self, imgcv):

        fname = str(uuid.uuid4())+'.jpg'
        imgpath = os.path.join( IMAGE_DIR_LOCAL, fname )
        cv2.imwrite(imgpath, imgcv)

        image = face_recognition.load_image_file(imgpath)
        face_locations = face_recognition.face_locations(image, model="hog")

        cindex = 0
        jsonresult = []
        for face in face_locations:
            cv2.rectangle(imgcv, (face[1], face[0]), (face[3], face[2]), COLORS[cindex], 2)
            cindex += 1
            if cindex > 15: cindex = 0

            jsonresult.append({
                'label': 'face',
                'score': '0',
                'boundingBox' : (
                    face[1], face[0],
                    face[3], face[2]
                )
            })

        face_encodings = face_recognition.api.face_encodings(image, face_locations, num_jitters=1)
        fi = 0
        for face in face_encodings:
            compare = face_recognition.api.compare_faces(self.mFaceEncodings, face, tolerance=0.6)
            facename = 'unknown'
            for i in range(len(self.mFaceEncodings)):
                if compare[i]:
                    facename = self.mFaceNames[i]
                    break
            jsonresult[fi]['label'] = facename
            fi += 1
            print(compare)

        return jsonresult


class Darkflow(metaclass=Singleton):

    mTFNet = None

    def __init__(self):
        options = {
            "model" : os.path.join(os.environ['MODEL_DIR'], 'darknet/cfg/yolo.cfg'),
            "load" : os.path.join(os.environ['MODEL_DIR'], 'darknet/bin/yolo.weights'),
            "threshold": 0.5
        }
        self.mTFNet = TFNet(options)
        print('Yolo object detector initialized')
        print()

    def run(self, imgcv):

        result = self.mTFNet.return_predict(imgcv)

        height, width, channels = imgcv.shape
        cindex = 0
        jsonresult = []
        for item in result:
            # Draw preview
            cv2.rectangle( imgcv,
                (item['topleft']['x'], item['topleft']['y']),
                (item['bottomright']['x'], item['bottomright']['y']), COLORS[cindex], 1 ) # Bounding box

            textSize = cv2.getTextSize( item['label'], CV2_FONTFACE, CV2_FONTSCALE, CV2_THICKNESS )
            print(textSize)
            clr = COLORS[cindex]
            cv2.rectangle( imgcv,
                (item['topleft']['x'], item['bottomright']['y']),
                (item['topleft']['x'] + textSize[0][0] + CV2_PADDING*2, item['bottomright']['y'] - textSize[0][1] - CV2_PADDING), clr, -1 ) # Text background
            cv2.putText( imgcv, item['label'],
                (item['topleft']['x'] + CV2_PADDING, item['bottomright']['y'] - CV2_PADDING),
                CV2_FONTFACE, CV2_FONTSCALE, (255-clr[0], 255-clr[1], 255-clr[2]), CV2_THICKNESS ) # Text
            cindex += 1
            if cindex > 15: cindex = 0

            jsonresult.append({
                'label' : item['label'],
                'score' : str(item['confidence']),
                'boundingBox' : (
                    item['topleft']['x'], item['topleft']['y'],
                    item['bottomright']['x'], item['bottomright']['y']
                ),
            })

        return jsonresult


class InceptionV3(metaclass=Singleton):

    OPTIONS = {
        'inceptionv3' : {
            'pre-trained' : {
                'graph' : os.path.join(os.environ['MODEL_DIR'], 'inceptionv3/imagenet/classify_image_graph_def.pb'),
                'label' : os.path.join(os.environ['MODEL_DIR'], 'inceptionv3/imagenet/imagenet_labels_sorted.txt'),
                'output' : 'softmax:0',
            },
            'flowers' : {
                'graph' : os.path.join(os.environ['MODEL_DIR'], 'inceptionv3/retrained/retrained_graph.pb'),
                'label' : os.path.join(os.environ['MODEL_DIR'], 'inceptionv3/retrained/retrained_labels.txt'),
                'output' : 'softmax:0',
            },
        }
    }

    mGraph = tf.Graph()

    def __init__(self):

        # Unpersists graph from file
        gpath = self.OPTIONS['inceptionv3']['pre-trained']['graph']
        with tf.gfile.FastGFile(gpath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            with self.mGraph.as_default():
                _ = tf.import_graph_def(graph_def, name='')
        tf.reset_default_graph()
        print('InceptionV3 classifier initialized')
        print()

    def run(self, imgcv):

        with tf.Session(graph=self.mGraph) as sess:

            # Loads label file, strips off carriage return
            now = -timestamp()
            lpath = self.OPTIONS['inceptionv3']['pre-trained']['label']
            label_lines = [line.rstrip() for line 
                            in tf.gfile.GFile(lpath)]
            print( 'load labels', now + timestamp() )

            # Feed the image_data as input to the graph and get first prediction
            now = -timestamp()
            tname = self.OPTIONS['inceptionv3']['pre-trained']['output']
            softmax_tensor = sess.graph.get_tensor_by_name(tname)
            print( 'get_tensor_by_name', now + timestamp() )
            
            # Read in the image_data
            #image_data = tf.gfile.FastGFile('/var/www/html/images/detection/ff8254fba9170f90652b87fcf1604a75.jpg', 'rb').read()
            now = -timestamp()
            image_data = cv2.imencode('.jpg', imgcv)
            print( 'imencode', now + timestamp() )
            now = -timestamp()
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data[1].tostring()})
            print( 'sess.run', now + timestamp() )
            
            # Sort to show labels of first prediction in order of confidence
            top_k = predictions[0].argsort()[-5:][::-1]
            
            jsonresult = []
            for node_id in top_k:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
                jsonresult.append({
                    'class' : human_string,
                    'score' : ' %.5f' % (score)
                })

            return jsonresult


    def writeGraphVisualize(self):
        
        with tf.Session(graph=self.mGraph) as sess:
            # `sess.graph` provides access to the graph used in a `tf.Session`.
            writer = tf.summary.FileWriter("/tmp/log/...", sess.graph)
            writer.close()


class ThreadingSimpleServer(ThreadingMixIn, HTTPServer):
    pass


class Handler(SimpleHTTPRequestHandler):

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', self.headers['Origin'])
        self.send_header('Access-Control-Allow-Methods', 'OPTIONS, GET, POST')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        print('something...')
        self.send_response(200)
        self.send_header('Content-type', 'text/plain;charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        if self.path == '/tfgraph':
            pass

        self.wfile.write(bytes('Arobot Imaging Services Server is running', "utf8"))

    def do_POST(self):

        print()
        print(self.path)
        
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

        ctype, pdict = cgi.parse_header(self.headers['Content-Type'])
        if ctype == 'multipart/form-data':
            # Parse multipart data using FieldStorage (so the code can be reused in cgi scripts)
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

                        if self.path=='/detection/yolo':
                            results = Darkflow().detect(imgcv)
                        elif self.path=='/classify/inceptionv3':
                            results = InceptionV3().detect(imgcv)
                        elif self.path=='/classify/darknet19':
                            pass
                        elif self.path=='/face/detect':
                            results = FaceRecognition().detect(imgcv)

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

        elif ctype == 'application/json':

            urlcomp = urllib.parse.urlparse(self.path)
            
            #ParseResult(scheme='', netloc='', path='/imaging', params='', query='key=API_KEY', fragment='')

            if urlcomp.path=='/imaging':

                responses = {
                    "responses" : []
                }

                requestsObj = json.loads( self.rfile.read(length).decode('utf8') )

                if 'requests' not in requestsObj:
                    self.send_response(400, "Bad request")
                    self.end_headers()
                    return

                for request in requestsObj['requests']:

                    # Validate format
                    if 'media' not in request:
                        self.send_response(400, "Attribute missing: media")
                        self.end_headers()
                        return
                    if 'services' not in request:
                        self.send_response(400, "Attribute missing: services")
                        self.end_headers()
                        return
                    if 'content' not in request['media'] and 'url' not in request['media']:
                        self.send_response(400, "Empty mandatory attribute: media")
                        self.end_headers()
                        return

                    result = {
                        "requestId" : request['requestId']
                    }

                    # Iterate through all services requested
                    for service in request['services']:

                        if 'type' not in service:
                            self.send_response(400, "Attribute missing: service type")
                            self.end_headers()
                            return

                        # Parse options
                        resultsLimit = 5
                        if 'options' in service:
                            options = service['options']
                            if 'resultsLimit' in options: resultsLimit = options['resultsLimit']

                        # Pre-process image
                        img = False
                        if 'content' in request['media']:
                            imgdata = b64decode( request['media']['content'] )
                            print(len(imgdata))
                            file_bytes = np.asarray( bytearray(imgdata), dtype=np.uint8 )
                            print(len(file_bytes))
                            img = cv2.imdecode(file_bytes, 1)
                            img = imgFit( img, 416 )
                        else:
                            pass

                        # Dispatching imaging task
                        if service['type']=='FACE':
                            result['faceEntities'] = FaceRecognition().run(img)
                        elif service['type']=='DETECT':
                            result['detectEntities'] = Darkflow().run(img)
                        elif service['type']=='CLASSIFY':
                            result['predictions'] = InceptionV3().run(img)

                        # Preview
                        fname = str(uuid.uuid4())+'.jpg'
                        outpath = os.path.join( IMAGE_DIR_LOCAL, fname )
                        cv2.imwrite(outpath, img)
                        result['preview'] = os.path.join( IMAGE_DIR_SERVE, fname )

                        responses['responses'].append( result )

                # Send /imaging response
                self.send_response(200)
                self.send_header('Content-type', 'application/json;charset=utf-8')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()

                print(responses)
                self.wfile.write(bytes(json.dumps(responses), "utf8"))
            
            else:
                self.send_response(404)
                return

        else:
            postdata = self.rfile.read(length).decode('utf8')
            json_response['header']['err_no'] = 400
            json_response['header']['err_msg'] = 'Only support Content-Type: multipart/form-data; boundary=...'
            json_response['debug'] = {
                'Content-Type' : ctype,
                'postdata' : json.loads( postdata ),
            }
            
        self.send_response(500)


server = ThreadingSimpleServer(('', PORT), Handler)
print("Serving HTTP traffic using port", PORT)
try:
    while 1:
        sys.stdout.flush()
        server.handle_request()
except KeyboardInterrupt:
    print("\nShutting down server per users request.")