import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn import linear_model
from sklearn.linear_model import LassoCV
from scipy.optimize import minimize
import scipy
import multiprocessing as mp

# import geopandas as gpd
import urllib
import datetime
import time
import warnings
import sklearn
from sklearn.metrics import r2_score


def getFlow(site,start,stop):
    url = 'https://waterdata.usgs.gov/nwis/dv?cb_00060=on&format=rdb&site_no=' + site + '&referred_module=sw&period=&begin_date='+start+'&end_date='+stop
    df = pd.read_csv(url, header=31, delim_whitespace=True)
    df.columns = ['usgs', 'site', 'datetime', 'q', 'a']
    df.index = pd.to_datetime(df.datetime)
    # get in ft/s by dividing by area, then convert to mm/day
    area = getArea(site)*2.59e12 # area in mm^2
    df = 2.447e12*df[['q']]/area
    df.q = df.q.astype(float, errors='ignore')
    df.columns = [site]
    return df

def getArea(site):
	link = 'https://waterdata.usgs.gov/nwis/inventory?agency_code=USGS&site_no=' + site
	webp=urllib.request.urlopen(link).read().decode()
	key = 'Drainage area: '
	idx = webp.find(key) + len(key)
	area = float(webp[idx:(idx+10)].split(' ')[0])
	return area

def a(t, alpha, v0):
	return alpha*v0*(1+t*v0)**(-1-alpha)
def fv(v,alpha,v0):
	return 1/(scipy.special.gamma(alpha)*v0)*np.exp(-v/v0)*(v/v0)**(alpha-1)

def get_rechargefit(x0,Q, positive=True):
	t = np.arange(len(Q))
	A = a(t, x0[0], x0[1])
	# create toeplitz matrix
	# performs convolution of IUH on input vector
	num_cols = len(Q)
	first_row = np.zeros(num_cols)
	first_row[0] = A[0]
	H = linalg.toeplitz(A, first_row)

	# validate over N folds  chosen at random indices
	N = 5
	traintest = []
	for i in range(N):
		frac = 1/N
		# only train/test over the second half of the dataset
		# allows for "spinup" to meet initial storage state
		idx_train = np.random.choice(range(int(len(Q)/2),len(Q)-1), int(len(Q)*frac), replace=False)
		idx_test = list(set(np.arange(int(len(Q)/2),len(Q))).difference(set(idx_train)))
		traintest.append((idx_train, idx_test))
	clf = linear_model.LassoCV(cv=traintest, positive=positive)
	clf.fit(H, Q)
	return clf, np.copy(H), np.copy(Q), t

def get_score(x0,Q, positive=True, logtransform=True):
	clf, H, Q, t = get_rechargefit(x0,Q,positive)
	rechargefit = clf.coef_
	Qfit = np.dot(H, rechargefit)
	idx = (Q>0)&(Qfit>0)
	Q = Q[idx]
	Qfit = Qfit[idx]
	if len(Qfit)==0:
		return 1e13
	else:
		if logtransform:
			return 1-r2_score(np.log(Q),np.log(Qfit))
		else:
			return 1-r2_score(Q,Qfit)

def getmin(args):
	f,x0,Q = args
	res = minimize(f, x0,method='Nelder-Mead', tol=1e-4, args = (Q))
	return res

def main():
	df = pd.read_csv('./flow.csv', index_col=0)
	sites = df.columns.values

	ress = []
	for col in df.columns:
		print(col)
		x0 = [0.5, 0.5]
		Q = df[col].values
		Q = Q[np.isfinite(Q)]
		idx = np.min([len(Q), 2000])
		Q = Q[-idx:]

		
		num = mp.cpu_count()
		p = mp.Pool(num)
		args = [(get_score,[np.random.uniform(low=0.2,high=1.5), np.random.uniform(low=0.2,high=1.5)],Q) for i in range(num)]
		res_candidates = p.map(getmin,args)
		idx = np.argmin([item.fun for item in res_candidates])
		res = res_candidates[idx]
		ress.append(res)

	    
	xs = [res.x for res in ress]
	tempdf = pd.DataFrame(xs)
	tempdf.index = df.columns
	tempdf.to_csv('ress.csv')

if __name__ == '__main__':
	main()





