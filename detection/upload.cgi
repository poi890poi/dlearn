#!/usr/bin/python3
# -*- coding: UTF-8 -*-# enable debugging

from string import Template

import cgi
import cgitb
cgitb.enable()

print("Content-Type: text/html;charset=utf-8")
print()

#cgi.print_directory()

with open('./templates/upload.html', 'r') as tfile:
    tpage = Template(tfile.read())
    print(tpage.safe_substitute(action='http://192.168.56.102:8000'))
