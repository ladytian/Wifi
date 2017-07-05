def kde(data):
    from sklearn.decomposition import PCA 
    from sklearn.neighbors import KernelDensity
    from sklearn.model_selection import GridSearchCV
    L = data.shape[0] #Read the length of the first dimension of the matrix
    if L < 10: 
        return

    (fill, ij, ap2idx, idx2ap, size) = wf2array.get_fill_ij(data['wflist'],30,convert_sig,False)

    # (fill, ij, ap2idx, idx2ap) = feature_selection((fill, ij), idx2ap, ap2idx)
    m = sp.csr_matrix((fill,ij),dtype=np.float32,shape=(size,ij[1,:].max() + 1)).todense()
    print data['wflist']
    print m

    #pca = PCA(n_components=15, whiten=True)
    #pcad_m = pca.fit_transform(m)

    params = {'bandwidth': np.logspace(-1, 1, 20)}
    params = {'bandwidth': [0.1]}
    grid = GridSearchCV(KernelDensity(), params)
    grid.fit(m)
    print 'bandwidth', grid.best_estimator_.bandwidth

    kde = grid.best_estimator_

    new_wifis = kde.sample(2, random_state=0)
    #new_wifis = pca.inverse_transform(new_wifis)
    for sigs in new_wifis:
        ptr = []
        for i in idx2ap:
            ap = '%012x' % (idx2ap[i])
            ptr.append(ap+';'+str(sigs[i]))
        print '|'.join(ptr)

def get_fill_ij(wflists,int averageApNum = 30,convertf = convertf_100,normed = True,ap2idx = None,idx2ap = None,fix = False,invalidap = None):
    cdef int i,size,dsize,fill_idx
    size = len(wflists) #shape[0]                                         
    cdef int ds = size
    cdef int maxsize = ds*averageApNum                                                                   
    
    cdef np.ndarray[np.float32_t,ndim=1] fill = np.zeros(ds * averageApNum ,dtype=np.float32)            
    cdef np.ndarray[np.int32_t,ndim=2] ij = np.zeros((2,ds*averageApNum),dtype=np.int32)
    
    dsize =  fill_idx = 0                                                                                
    if ap2idx is None:
        ap2idx = dict()                                                                                  
        fix = False
    if idx2ap is None:                                                                                   
        idx2ap = dict()                                                                                  
        fix = False
    for i in xrange(size):                                                                               
        wf = str_to_wf(wflists[i],convertf,normed)                                                      
        for mac in wf:
            if invalidap is not None and mac in invalidap:                                               
                continue 
            if mac not in ap2idx:                                                                        
                if fix:
                    continue                                                                             
                ap2idx[mac] = dsize                                                                      
                idx2ap[dsize] = mac                                                                      
                dsize += 1
                    
            if fill_idx >= maxsize:                                                                      
                logging.warning('exceed max fill size:[%d] [%d],ignoring' % (fill_idx,maxsize) )         
                continue
            ij[0,fill_idx] = i                                                                           
            ij[1,fill_idx] = ap2idx[mac]#wfs[i,1+j*2]                                                    
                
            fill[fill_idx] = wf[mac]                                                                     
            fill_idx += 1
            
    return fill[:fill_idx],ij[:,:fill_idx],ap2idx,idx2ap,size