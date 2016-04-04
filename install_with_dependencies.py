# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 16:52:45 2016

@author: miles
"""

import pip

def install(package):
	pip.main(['install',package ])

def upgrade(package):
	pip.main(['install',"--upgrade",package ])

# Example
if __name__ == '__main__':
    install("music21")
    install("sounddevice")
    upgrade("https://github.com/pabloriera/pymam/tarball/master")
    