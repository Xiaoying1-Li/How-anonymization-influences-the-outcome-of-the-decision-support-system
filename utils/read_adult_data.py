#!/usr/bin/env python
# coding=utf-8

from models.gentree import GenTree
from models.numrange import NumRange
from utils.utility import cmp_str
import functools
import pickle

import pdb

ATT_NAMES = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target","index",]
# 8 attributes are chose as QI attributes
# age and education levels are treated as numeric attributes
# only matrial_status and workclass has well defined generalization hierarchies.
# other categorical attributes only have 2-level generalization hierarchies.
QI_INDEX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
IS_CAT = [False, True, True, False, False, True, True, False, True, False, True, True, True]
SA_INDEX = [13, 14]

__DEBUG = False


def read_data():
    """
    read microda for *.txt and return read data
    """
    QI_num = len(QI_INDEX)
    data = []
    numeric_dict = []
    for i in range(QI_num):
        numeric_dict.append(dict())
    # oder categorical attributes in intuitive order
    # here, we use the appear number
    with open('data/process/heart_disease_a.csv', 'r') as data_file:
        for line_num, line in enumerate(data_file):
            if line_num == 0:  # 跳过第一行
                continue
            line = line.strip()
            # remove empty and incomplete lines
            # only 30162 records will be kept
            if len(line) == 0 or '?' in line:
                continue
            # remove double spaces
            line = line.replace(' ', '')
            temp = line.split(',')
            ltemp = []
            for i in range(QI_num):
                index = QI_INDEX[i]
                if IS_CAT[i] is False:
                    try:
                        numeric_dict[i][temp[index]] += 1
                    except KeyError:
                        numeric_dict[i][temp[index]] = 1
                ltemp.append(temp[index])
            ltemp.append(';'.join([temp[SA_INDEX[0]], temp[SA_INDEX[1]]]))
            data.append(ltemp)
    # pickle numeric attributes and get NumRange
    for i in range(QI_num):
        if not IS_CAT[i]:
            with open('data/hierarchy/heart_' + ATT_NAMES[QI_INDEX[i]] + '_static.pickle', 'wb') as static_file:
                sort_value = list(numeric_dict[i].keys())
                sort_value.sort(key=functools.cmp_to_key(cmp_str))
                pickle.dump((numeric_dict[i], sort_value), static_file)
    return data


def read_tree():
    """read tree from data/tree_*.txt, store them in att_tree
    """
    att_names = []
    att_trees = []
    for t in QI_INDEX:
        att_names.append(ATT_NAMES[t])
    for i in range(len(att_names)):
        if IS_CAT[i]:
            att_trees.append(read_tree_file(att_names[i]))
        else:
            att_trees.append(read_pickle_file(att_names[i]))
    return att_trees


def read_pickle_file(att_name):
    """
    read pickle file for numeric attributes
    return numrange object
    """
    try:
        static_file = open('data/hierarchy/heart_' + att_name + '_static.pickle', 'rb')
        (numeric_dict, sort_value) = pickle.load(static_file)
    except:
        print ("Pickle file not exists!!")
    static_file.close()
    result = NumRange(sort_value, numeric_dict)
    return result


def read_tree_file(treename):
    """read tree data from treename
    """
    leaf_to_path = {}
    att_tree = {}
    prefix = 'data/hierarchy/heart_'
    postfix = ".txt"
    treefile = open(prefix + treename + postfix, 'r')
    att_tree['*'] = GenTree('*')
    if __DEBUG:
        print("Reading Tree" + treename)
    for line in treefile:
        # delete \n
        if len(line) <= 1:
            break
        line = line.strip()
        temp = line.split(';')
        # copy temp
        temp.reverse()
        for i, t in enumerate(temp):
            isleaf = False
            if i == len(temp) - 1:
                isleaf = True
            # try and except is more efficient than 'in'
            try:
                att_tree[t]
            except:
                att_tree[t] = GenTree(t, att_tree[temp[i - 1]], isleaf)
    if __DEBUG:
        print ("Nodes No. = %d" % att_tree['*'].support)
    treefile.close()
    return att_tree