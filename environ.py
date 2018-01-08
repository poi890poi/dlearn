#!/usr/bin/python3
# -*- coding: UTF-8 -*-# enable debugging
import cgitb, cgi
cgitb.enable()
print("Content-Type: text/html;charset=utf-8")
print()
cgi.print_environ()
cgi.print_environ_usage()
