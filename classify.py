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

import cgitb
import cgi
cgitb.enable()

def md5digest(raw):
    m = md5()
    m.update(raw)
    return m.hexdigest()

print("Content-Type: text/html;charset=utf-8")
print()
print("Invoking script")
print(sys.stdout.encoding)
print(tempfile.gettempdir())

now = timestamp()

fs = cgi.FieldStorage()
if 'userfile' in fs:
    fileitem = fs['userfile']
    if fileitem.file:
        data = fileitem.file.read()
        outpath = os.path.join(tempfile.gettempdir(), md5digest(data))
        with open(outpath, 'wb') as fout:
            print(outpath)
            fout.write(data)
            #shutil.copyfileobj(fileitem.file, fout, 100000)
            #exec(open("./classify_image.py --image_file "+outpath).read())
            #cscript = open("./classify_image.py").read()
            #exec(cscript)
            stdoutdata = subprocess.getoutput("python3 classify_image.py --image_file "+outpath)
            print()
            print("stdoutdata: " + stdoutdata)

print()
print(timestamp()-now)

#exec(open("../classify_image.py").read())