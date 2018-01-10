#!/usr/bin/python3
# -*- coding: UTF-8 -*-# enable debugging

import sys
import numpy as np

from hashlib import md5
import tempfile
import os
import json
import urllib.request
from urllib.error import URLError
import http.client, urllib.parse

import time
timestamp = lambda: int(round(time.time() * 1000))

import cv2

import cgitb
import cgi
cgitb.enable()

CLASSIFY_DIMENSION = 299
CLASSIFY_ENDPOINT = '192.168.56.102:8000'
CLASSIFY_FORMATS = ('image/jpeg', 'image/png', 'image/webp')
IMAGE_DIR = '/var/www/html/images/classify/'
IMAGE_DIR_MAPPED = 'http://192.168.56.102/images/classify/'

def md5digest(raw):
    m = md5()
    m.update(raw)
    return m.hexdigest()


def resize_n_pad( fileobj, save = False ):

    # Load as opencv image object
    imdata = fileobj.read()
    file_bytes = np.asarray( bytearray( imdata ), dtype=np.uint8 )
    img = cv2.imdecode( file_bytes, 1 )

    # Resize and pad image
    height, width, channels = img.shape
    # Resize
    if width >= height: # Landscape
        height = int(CLASSIFY_DIMENSION * height / width)
        width = CLASSIFY_DIMENSION
    else: # Portrait
        width = int(CLASSIFY_DIMENSION * width / height)
        height = CLASSIFY_DIMENSION
    resized_image = cv2.resize(img, (width, height))
    # Padding
    if width < CLASSIFY_DIMENSION: # Too wide
        bleft = int((CLASSIFY_DIMENSION-width)/2)
        bright = CLASSIFY_DIMENSION - bleft - width
        resized_image = cv2.copyMakeBorder(resized_image, top=0, bottom=0, left=bleft, right=bright, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
    elif height < CLASSIFY_DIMENSION: # Too tall
        btop = int((CLASSIFY_DIMENSION-height)/2)
        bbottom = CLASSIFY_DIMENSION - btop - height
        resized_image = cv2.copyMakeBorder(resized_image, top=btop, bottom=bbottom, left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )

    imdata = cv2.imencode('.jpg', resized_image)
    if imdata[0]:
        # Save for preview (internal testing)
        if save:
            fname = md5digest(imdata[1].tostring()) + '.jpg'
            outpath = os.path.join( IMAGE_DIR, fname )
            with open(outpath, 'wb') as f:
                f.write(imdata[1].tostring())
            return ( 'url', os.path.join(IMAGE_DIR_MAPPED, fname) )
        else:
            return ( 'bin', imdata[1].tostring() )

    return ('', '')


def classify():
    
    fs = cgi.FieldStorage()
    
    if fs.list:
        for item in fs.list:
            if item.file and item.type in CLASSIFY_FORMATS:
                img = resize_n_pad(item.file, True) # Request with image url
                #img = resize_n_pad(item.file, False) # Request with image binary data
                headers = {"Content-type": "application/x-www-form-urlencoded",
                            "Accept": "text/plain",
                            "Arobot-Data-Type": img[0],
                            "Arobot-Request-Id": item.name}
                conn = http.client.HTTPConnection(CLASSIFY_ENDPOINT)
                if img[0]=='url':
                    conn.request("POST", "", img[1], headers)
                    response = conn.getresponse()
                    print( response.read().decode('utf-8') )
                    return
                elif img[0]=='bin':
                    conn.request("POST", "", img[1], headers)
                    response = conn.getresponse()
                    print( response.read().decode('utf-8') )
                    return
                else:
                    pass

    json_response = {
        'header' : {
            'api' : 'image_classify_wrapped',
            'date' : '2018-01-10',
            'err_no' : 400,
            'err_msg' : 'No valid data in multipart/form-data'
        },
    }
    print( json.dumps(json_response) )

print("Content-Type: application/json;charset=utf-8")
print() # End of HTTP response headers

classify()


#exec(open("../classify_image.py").read())