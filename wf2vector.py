import csv
import numpy as np
from sklearn.feature_extraction import DictVectorizer

#dt1 = np.dtype([('grid', 'S23'), ('tag', 'S10'), ('x', 'f4'), ('y', 'f4'), ('wflist' ,'S600')])
#dt2 = np.dtype([('grid', 'S23'), ('id', 'S20'), ('tag', 'S10'), ('x', 'f4'), ('y', 'f4'), ('wflist' ,'S600')])

file = r"D:\workspace\Wifi\data\test.in"

def get_data(file):
	data = []
	for line in open(file):
		line = line.strip('\n').split('\t')
		data.append(line)

	return data


def get_id_wflist(d, id):
	wflist = []
	for info in d:
		if info[1] != id:
			continue
		elif info[1] == id:
			dict_wf = {}
			wfs = info[5].strip().split('|')
			for wf in wfs:
				wf = wf.split(';')
				if len(wf) != 3:
					continue
				dict_wf[wf[0]] = int(wf[1])
			wflist.append(dict_wf)

	return wflist


def dict_vector(D):
	v = DictVectorizer(sparse=False)
	X = v.fit_transform(D)

	return X


def get_idlist(d):
	idlist = []
	for info in d:
		if info[1] not in idlist:
			idlist.append(info[1])

	return idlist


if __name__ == '__main__':
	data = get_data(file)
	idlist = get_idlist(data)
	print 'len = ' + str(len(idlist))

	for id in idlist:
		dict_wf = get_id_wflist(data, id)
		v = dict_vector(dict_wf)
		print v.shape
		print v