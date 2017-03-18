import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import seaborn as sns
sns.set_style("whitegrid")

from collections import namedtuple  
import sys
import os
import math
import operator
import random
import linecache

wifi_df = pd.read_csv(r"D:\workspace\Wifi\all.csv")
#wifi_df.info()
#print wifi_df.describe()

def parse_wf(mode, nfeature, wfs):
	if mode == 1:
		wfs = smoothwf(wfs, nfeature)
	# print wfs, 'wf'
	wf_dict = {}
	wfs = wfs.strip().split('|')
	for wf in wfs:
		wf = wf.split(';')
		if len(wf) != 3:
			continue
		wf_dict[wf[0]] = int(wf[2])
	return wf_dict

def smoothwf(wfs, nfeature):
	wfs = wfs.split('|')
	wf_dict = {}
	for wf in wfs:
		wf = wf.split(';')
		if len(wf) != 3:
			continue
		wf_dict[wf[0]+'\t'+wf[1]] = wf[2]
	sorted_wf = sorted(wf_dict.items(), key=operator.itemgetter(1), reverse=False)
	ret_wfs = ""
	for wf in sorted_wf[:nfeature]:
		tmp = wf[0].split('\t')
		ret_wfs += tmp[0]+';'+tmp[1]+';'+wf[1]+'|'
	return ret_wfs
def getrandomline(filename):
	count = len(open(filename, 'r').readlines())
	x = random.randint(1, count)
	#print 'randint x = %d' % x
	return parse_wf(0, 2, linecache.getline(filename, x))


def merge_wifi(ap_name):
	switcher = {
	"28A-101": "2016222_28-A101",
	"28A-102": "2016222_28-A102",
	"28A-103": "2016222_28-A103",
	"28A-104": "2016222_28-A104",
	"28A-105": "2016222_28-A105",
	"28A-106": "2016222_28-A106",
	"28A-201": "2016222_28-A201",
	"28A-202": "2016222_28-A202",
	"28A-203": "2016222_28-A203",
	"28A-204": "2016222_28-A204",
	"28A-205": "2016222_28-A205",
	"28A-206": "2016222_28-A206",
	"28A-207": "2016222_28-A207",
	"28A-208": "2016222_28-A208",
	"28A-209": "2016222_28-A209",
	"28A-210": "2016222_28-A210",
	}
	wf = getrandomline('./wifiscanner/' + switcher.get(ap_name, "nothing"))
	s = ""
	for key, value in wf.items():
		s = s + str(key) + ";" + str(value) + "|"

	return s[:len(s)-1]

ap = wifi_df["ap_name"]
wifi_df["wifi_fingerprinter"] = " "

for i in range(0, 387510):
	if i%1000:
		print i
	wifi_df.ix[[i],["wifi_fingerprinter"]] = merge_wifi(ap[i])

wifi_df.to_csv('a.csv', index=False)