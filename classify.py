#!/usr/bin/python3
# -*- coding: UTF-8 -*-# enable debugging

import sys

from hashlib import md5
import tempfile
import shutil
import os
import subprocess

import time
timestamp = lambda: int(round(time.time() * 1000))

import cv2
import numpy

import cgitb
import cgi
cgitb.enable()

def md5digest(raw):
    m = md5()
    m.update(raw)
    return m.hexdigest()

def classify():
    
    fs = cgi.FieldStorage()
    
    # Validate data
    print(fs.type)
    print(fs.headers)
    if 'userfile' not in fs: return 0
    len = int(fs.headers['content-length'])
    if len > 4*1048576: return 0
    fileitem = fs['userfile']
    if not fileitem.file: return 0
    
    # Load as opencv image object
    data = fileitem.file.read()
    file_bytes = numpy.asarray( bytearray( data ), dtype=numpy.uint8 )
    img = cv2.imdecode( file_bytes, 1 )
    fname = md5digest(data) + '.jpg'
    outpath = os.path.join( '/var/www/html/images/', fname )
    
    # Resize and pad image
    cdim = 299
    height, width, channels = img.shape
    print(width, height, channels)
    if width >= height: # Landscape
        height = int(cdim * height / width)
        width = cdim
    else:
        width = int(cdim * width / height)
        height = cdim
    print(width, height, channels)
    resized_image = cv2.resize(img, (width, height))
    if width < cdim:
        bleft = int((cdim-width)/2)
        bright = cdim - bleft - width
        resized_image = cv2.copyMakeBorder(resized_image, top=0, bottom=0, left=bleft, right=bright, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
    elif height < cdim:
        btop = int((cdim-height)/2)
        bbottom = cdim - btop - height
        resized_image = cv2.copyMakeBorder(resized_image, top=btop, bottom=bbottom, left=0, right=0, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
    outpath = outpath+'.jpg'
    cv2.imwrite(outpath, resized_image)

    # Do classify using Inception V3 tutorial script
    stdoutdata = subprocess.getoutput("python3 classify_image.py --image_file "+outpath)
    print()
    print("stdoutdata: " + stdoutdata)

print("Content-Type: text/html;charset=utf-8")
print()
print("Invoking script")
print(sys.stdout.encoding)
print(tempfile.gettempdir())
#print(cv2)

now = timestamp()

classify()

print()
print(timestamp()-now)

#exec(open("../classify_image.py").read())