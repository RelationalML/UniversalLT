import numpy as np
import matplotlib.pylab as plt

def relu(x):
    return np.clip(x, a_min=0, a_max=None)

def exhaustive5(target, numbers):
    n = len(numbers)
    if n > 30:
        numbers = numbers[:30]
    #add zero to numbers or run greedy for one element before
    numbers = np.concatenate([numbers, np.zeros(4)])
    n = len(numbers)
    err = np.abs(target)
    indVec = np.zeros(5,dtype="int")+int(n)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    for m in range(l+1, n):
                        ind = np.array([i,j,k,l,m],dtype="int")
                        diff = np.abs(target-np.sum(numbers[ind]))
                        if diff < err:
                            err = diff
                            indVec = ind
    if np.min(indVec) < n:
        return np.sum(numbers[indVec]), indVec[indVec < n-4]
    else:
        return 0, np.array([np.NAN])

def exhaustiveLarge(target, numbers):
    ind = np.argsort(numbers)
    if target > 0:
        ind = np.flip(ind)
    test = np.array([np.sum(numbers[ind[:i]]) for i in range(len(ind))])
    imin = np.argmin(np.abs(test-target))
    ind = ind[:imin]
    sum = np.sum(numbers[ind])
    indRemain = np.setdiff1d(np.arange(len(numbers)), ind)
    #print(ind)
    numbers = np.delete(numbers, ind)
    sumNew, indNew = exhaustive5(target-sum, numbers)
    #print(indNew)
    if np.isnan(indNew[0]):
        return sum, ind
    else:
        return sum+sumNew, np.concatenate([ind, indRemain[indNew]])

def exhaustive(target, numbers):
    if np.abs(target) < 4:
        sum, ind = exhaustive5(target, numbers)
    else:
        sum, ind = exhaustiveLarge(target, numbers)
    return sum, ind

def subsetsum2(x, s):
    sfull = s.copy()
    #s = np.random.uniform(-1,1,n)
    n = len(s)
    sum = 0
    sum, el, s = addGreedy(x, sum, s)
    #print(el)
    while el != 0:
        sum1, s1 = subsetsum(x-sum, s)
        err1 = np.abs(sum1+sum-x)
        err = np.abs(x-sum)
        print(err)
        print(err1)
        if err > err1:
            sum = sum1+sum
            s = s1
        else:
            el = 0
        #print(el)
    #print(np.abs(sum-x))
    print(sum)
    print(n-len(s))
    return np.abs(sum-x)

def subsetSumObj(y, mu, s, m):
    mask = 1/(1+np.exp(-m))
    return ((y-np.sum(s*mask))**2 - mu*np.sum(m**2))*0.5

def subsetSumGrad(y, mu, s, m):
    mask = 1/(1+np.exp(-m))
    grad = (np.sum(s*mask)-y)*s*mask*(1-mask) - mu*m
    return grad

def cut(x):
    for i in range(len(x)):
        if x[i] > 2:
            x[i] = 10
        if x[i] < -2:
            x[i] = -10
    return x

#prune univariate function
def prune_first_balanced(c1, c2, Nlog, N1, ff):
    #first layer
    b = np.random.uniform(-1,1,N1)
    w1 = np.random.uniform(-2,2,N1)
    w2 = np.random.uniform(-2,2, [N1, Nlog])
    tb = np.linspace(c1,c2,num=Nlog+1)
    m = (ff(tb[1:])-ff(tb[:(-1)]))/(tb[1:]-tb[:(-1)])
    tw = np.zeros(Nlog)
    tw[0] = np.abs(m[0])
    tw[1:] = np.sqrt(np.abs(m[1:]-m[:(-1)]))
    tb = tb[:Nlog]
    win = np.zeros(Nlog)
    ipr = np.where((w1>0)&(b<0))[0]
    indw = np.zeros(1)
    #preference for larger w rather than smaller
    #i = 0 with positive win
    ipr0 = np.where((w1>0)&(w2[:,0]>0))[0]
    ii = np.argmin(np.abs(w1[ipr0]*w2[ipr0,0]-tw[0]))
    ii = ipr0[ii]
    win[0] = w1[ii]*w2[ii,0]
    indw[0] = ii
    ipr = np.setdiff1d(ipr, ii)
    for i in range(1,Nlog):
        if len(ipr) < 10:
            ipr = np.where(w1>0)[0]
            ipr = np.setdiff1d(ipr, indw)
        ii = np.argmin(np.abs(np.abs(w1[ipr]*w2[ipr,i])-tw[i]))
        ii = ipr[ii]
        win[i] = w1[ii]*w2[ii,i]
        if np.abs(np.abs(win[i])-tw[i]) < 0.5:
            ipr = np.setdiff1d(ipr, ii)
            indw = np.concatenate([indw, np.array([ii])])
        else:
            s = w1[ipr]*w2[ipr,0]
            ns = len(s)
            win[i], indUsed = exhaustive(tw[i], s)
            indw = np.concatenate([indw, ipr[indUsed]])
            ipr = np.delete(ipr, indUsed)
    usedParam = 2*len(indw)
    #target bias
    tb = -tb*win
    #prune b
    ind = np.where(b>0)[0]
    ind = np.setdiff1d(ind, indw)
    b = b[ind]
    w2 = w2[ind,:]
    bb = np.zeros(Nlog)
    for i in range(1,Nlog):
        s = b*w2[:,i]
        ns = len(s)
        bb[i], indUsed = exhaustive(tb[i], s)
        b = np.delete(b,indUsed)
        w2 = np.delete(w2,indUsed,axis=0)
        usedParam = usedParam + 2*len(indUsed)
    print("# used parameters: ", usedParam)
    return win, bb

def prune_second(win, bb, N, ff):
    #linear combination of relus: define targets wout, blast
    st = -bb/np.where(win==0,1,win)
    ind = np.argsort(st)
    st = st[ind]
    bb = bb[ind]
    win = win[ind]
    Nf = len(bb)
    st = np.concatenate([st, np.array([2*st[-1]-st[-2]]).reshape(-1)])
    y = ff(st)
    m = np.array([(y[i+1]-y[i])/(st[i+1]-st[i]) for i in range(Nf)])
    wout = np.zeros(Nf)
    wout[1:] = (m[1:]-m[:(-1)])
    wout[0] = m[0]+np.sum(np.where(win<0,1,0)*wout)
    wout = wout/np.abs(np.where(win==0,1,win))
    blast = ff(st[0])-np.sum(relu(st[0]*win+bb)*wout)
    #prune towards targets
    w1 = np.random.uniform(-2,2,N)
    w2 = np.random.uniform(-2,2,N)
    b = np.random.uniform(-1,1,N)
    #relu input is always positive -> w1 (or b) need to be positive to send signal to the next layer
    #prune last layer bias
    indB = np.where((b>0)&(w1<0))[0]
    s = b[indB]*w2[indB]
    #print(s)
    if np.abs(blast) < 0.00001:
        bl = 0
    else:
        bl, indB = exhaustive(blast, s)
    if np.abs(blast-bl) > 0.001:
        indB = np.where((b>0))[0]
        s = b[indB]*w2[indB]
        bl, indB = exhaustive(blast, s)
        ind = np.setdiff1d(np.arange(N), indB)
        w1 = w1[ind]
        w2 = w2[ind]
    usedParam = 2*len(indB)
    #prune weights
    ind = np.where(w1>0)[0]
    w1 = w1[ind]
    w2 = w2[ind]
    ww = np.zeros(Nf)
    #obtain target weights
    for i in range(Nf):
        s = w1*w2
        ns = len(s)
        ww[i], indUsed = exhaustive(wout[i], s)
        indUsed = indUsed[indUsed<ns]
        w1 = np.delete(w1,indUsed)
        w2 = np.delete(w2,indUsed)
        usedParam = usedParam + 2*len(indUsed)
    print("# used parameters: ", usedParam)
    return win, bb, ww, bl

def univNet(x, win, wout, bin, bout):
    return np.sum(wout*relu(win*x + bin)) + bout

def prune_lin(target, N):
    w1 = np.random.uniform(-2,2,N)
    w2 = np.random.uniform(-2,2,N)
    k = len(target)
    proxy = np.zeros(k)
    err = 0
    usedParam = 0
    for i in range(k):
        indLoc = np.where(w1*np.sign(target[i]) > 0)[0]
        s = w1[indLoc]*w2[indLoc]
        ns=len(s)
        proxy[i], indUsed = exhaustive(target[i], s)
        indUsed = indUsed[indUsed<ns]
        w1 = np.delete(w1,indUsed)
        w2 = np.delete(w2,indUsed)
        usedParam = usedParam + 2*len(indUsed)
        errLoc = np.abs(target[i]-proxy[i])
        if errLoc > err:
            err = errLoc
    print("# used parameters: ", usedParam)
    return err, proxy

def prune_univariate(c1, c2, N, width1, width2, ftarget):
    #prune ftarget
    win, bb = prune_first_balanced(c1, c2, N, width1, ftarget)
    win, bb, wout, blast = prune_second(win, bb, width2, ftarget)
    #output neural net
    fout = lambda x: univNet(x, win, wout, bb, blast)
    #error assessment
    domain = np.linspace(c1,c2,num=10000)
    yprox = np.array([fout(x) for x in domain])
    yy = ftarget(domain)
    err = np.max(np.abs(yprox-yy))
    # plt.plot(domain,yy)
    # plt.plot(domain,yprox)
    # plt.show()
    return err, fout


def prune_poly(Nlog, Nexp, expVec, width):
    #prune log(1+x)
    errLog, fLog = prune_univariate(0,1, Nlog, width[0], width[1], lambda x: np.log(0.5*(1+x)))
    print("err log(1/2(1+x))")
    print(errLog)
    #prune exponents
    errExp, expProx = prune_lin(expVec, width[2])
    print("err Exponent")
    print(errExp)
    #prune exp(x)
    c = np.max(expVec)*np.log(2)
    errExp, fExp = prune_univariate(-c-0.07, 0.0, Nexp, width[3], width[4], lambda x: np.exp(x))
    print("err exp(x)")
    print(errExp)
    #combined error
    domain = np.linspace(0,1,num=10000)
    k = len(expVec)
    err = np.zeros(k)
    errRel = np.zeros(k)
    for j in range(k):
        yprox = np.array([fExp(expProx[j]*fLog(x)) for x in domain])
        yy = (0.5*(1+domain))**expVec[j]
        err[j] = np.max(np.abs(yprox-yy))
        errRel[j] = np.max(np.abs(yprox-yy)/yy)
    #     plt.plot(domain, yprox)
    #     plt.plot(domain, yy)
    # plt.show()
    return err, errRel

np.random.seed(seed=42)
Nlog = 10
Nexp = 40
width = np.array([200,200,200,500,500])
expVec = np.array([1,2,3,4])
err, errRel = prune_poly(Nlog, Nexp, expVec, width)
print("Pruning error for polynomials:")
print(err)
#total parameters after pruning 1126, before pruning: 431800, density of 0.0026

#pruning error for sin(2*pi*x)
print("Pruning error for sin:")
errLog, _ = prune_univariate(0, 1, 21, 250, 250, lambda x: np.sin(2*np.pi*x))
print(errLog)
#total parameters after pruning 436, before pruning: 68292, density: 0.0064
