#!/usr/bin/python3
# -*- coding: UTF-8 -*-# enable debugging

import sys

import cgitb
import cgi
cgitb.enable()

print("Content-Type: text/html;charset=utf-8")
print()
print("Invoking script")
print(sys.stdout.encoding)

fs = cgi.FieldStorage()
for value in fs:
    print(value)

#exec(open("../classify_image.py").read())