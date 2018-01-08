#!/usr/bin/python3
# -*- coding: UTF-8 -*-# enable debugging

import sys
import numpy as np

from hashlib import md5
import tempfile
import os
import json
import urllib.request

import time
timestamp = lambda: int(round(time.time() * 1000))

import cv2

import cgitb
import cgi
cgitb.enable()

CLASSIFY_DIMENSION = 299

def md5digest(raw):
    m = md5()
    m.update(raw)
    return m.hexdigest()

def classify():
    
    fs = cgi.FieldStorage()
    
    # Validate data
    json_response['input']['fs_type'] = fs.type
    json_response['input']['fs_headers'] = fs.headers
    if 'userfile' not in fs: return 0
    len = int(fs.headers['content-length'])
    if len > 4*1048576: return 0
    fileitem = fs['userfile']
    if not fileitem.file: return 0
    
    data = fileitem.file.read()
    fname = md5digest(data) + '.jpg'

    # Load as opencv image object
    file_bytes = np.asarray( bytearray( data ), dtype=np.uint8 )
    img = cv2.imdecode( file_bytes, 1 )
    outpath = os.path.join( '/var/www/html/images/', fname )
    
    # Resize and pad image
    height, width, channels = img.shape
    json_response['input']['dim_original'] = (width, height, channels)
    
    # Resize
    if width >= height: # Landscape
        height = int(CLASSIFY_DIMENSION * height / width)
        width = CLASSIFY_DIMENSION
    else: # Portrait
        width = int(CLASSIFY_DIMENSION * width / height)
        height = CLASSIFY_DIMENSION
    json_response['input']['dim_transformed'] = (width, height, channels)
    resized_image = cv2.resize(img, (width, height))

    # Padding
    if width < CLASSIFY_DIMENSION:
        bleft = int((CLASSIFY_DIMENSION-width)/2)
        bright = CLASSIFY_DIMENSION - bleft - width
        resized_image = cv2.copyMakeBorder(resized_image, top=0, bottom=0, left=bleft, right=bright, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
    elif height < CLASSIFY_DIMENSION:
        btop = int((CLASSIFY_DIMENSION-height)/2)
        bbottom = CLASSIFY_DIMENSION - btop - height
        resized_image = cv2.copyMakeBorder(resized_image, top=btop, bottom=bbottom, left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )

    cv2.imwrite(outpath, resized_image)
    json_response['input']['file_name'] = outpath

    with urllib.request.urlopen('http://192.168.56.102:8000/'+fname) as response:
      json_response['response'] = json.loads(response.read().decode('utf-8'))


print("Content-Type: application/json;charset=utf-8")
print() # End of HTTP response headers

json_response = {
    'env' : {},
    'input' : {},
    'profiling' : {},
    'predictions' : [],
}
json_response['env']['encoding'] = sys.stdout.encoding
json_response['env']['opencv'] = cv2.__version__

json_response['profiling']['total'] = -timestamp()

classify()

json_response['profiling']['total'] += timestamp()
print( json.dumps(json_response) )

#exec(open("../classify_image.py").read())