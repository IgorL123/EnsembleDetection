


import os
import xml.etree.ElementTree as ET
import codecs
import cv2
import sys
import numpy as np
import random
import shutil
import shapely.geometry as shgeo
import re
import pickle
import math
import copy


LABELS = [{'name':'plane', 'id': 0}, {'name':'ship', 'id':1}, {'name':'storage tank', 'id':2}, {'name':'baseball diamond', 'id':3}, {'name': 'tennis court', 'id':4}, {'name':'basketball court', 'id':5}, {'name':'ground track field', 'id':6}, {'name':'harbor', 'id':7}, {'name':'bridge', 'id':8}, {'name':'large vehicle', 'id':9}, {'name':'small vehicle', 'id':10}, {'name':'helicopter', 'id':11}, {'name':'roundabout', 'id':12}, {'name':'soccer ball field', 'id':13}, {'name':'swimming pool', 'id':14}, {'name':'container crane', 'id':15}]


def distance(point1, point2):
    return np.sqrt(np.sum(np.square(point1 - point2)))

small_count = 0
def parse_bod_poly(filename):
    objects = []
    print('filename:', filename)
    f = []
    if (sys.version_info >= (3, 5)):
        fd = open(filename, 'r')
        f = fd
    elif (sys.version_info >= 2.7):
        fd = codecs.open(filename, 'r')
        f = fd
    while True:
        line = f.readline()
        if line:
            splitlines = line.strip().split(' ')
            object_struct = {}
            ### clear the wrong name after check all the data
            #if (len(splitlines) >= 9) and (splitlines[8] in classname):
            if (len(splitlines) < 9):
                continue
            if (len(splitlines) >= 9):
                    object_struct['name'] = splitlines[8]
            if (len(splitlines) == 9):
                object_struct['difficult'] = '0'
            elif (len(splitlines) >= 10):
                #print 'splitline 9: ', splitlines[9]
                # if splitlines[9] == '1':
                if (splitlines[9] == 'tr'):
                    object_struct['difficult'] = '1'
                    #print '<<<<<<<<<<<<<<<<<<<<<<<<<<'
                else:
                    object_struct['difficult'] = splitlines[9]
                    #print '!!!!!!!!!!!!!!!!!!'
                # else:
                #     object_struct['difficult'] = 0
            object_struct['poly'] = [(float(splitlines[0]), float(splitlines[1])),
                                     (float(splitlines[2]), float(splitlines[3])),
                                     (float(splitlines[4]), float(splitlines[5])),
                                     (float(splitlines[6]), float(splitlines[7]))
                                     ]
            gtpoly = shgeo.Polygon(object_struct['poly'])
            object_struct['area'] = gtpoly.area
            poly = list(map(lambda x:np.array(x), object_struct['poly']))
            object_struct['long-axis'] = max(distance(poly[0], poly[1]), distance(poly[1], poly[2]))
            object_struct['short-axis'] = min(distance(poly[0], poly[1]), distance(poly[1], poly[2]))
            if (object_struct['long-axis'] < 15):
                object_struct['difficult'] = '1'
                global small_count
                small_count = small_count + 1
            objects.append(object_struct)
        else:
            break
    return objects

def dots4ToRec4(poly):
    xmin, xmax, ymin, ymax = min(poly[0][0], min(poly[1][0], min(poly[2][0], poly[3][0]))), \
                            max(poly[0][0], max(poly[1][0], max(poly[2][0], poly[3][0]))), \
                             min(poly[0][1], min(poly[1][1], min(poly[2][1], poly[3][1]))), \
                             max(poly[0][1], max(poly[1][1], max(poly[2][1], poly[3][1])))
    return xmin, ymin, xmax, ymax


def parse_bod_rec(filename):
    objects = parse_bod_poly(filename)
    for obj in objects:
        poly = obj['poly']
        bbox = dots4ToRec4(poly)
        obj['bndbox'] = bbox
    return objects


