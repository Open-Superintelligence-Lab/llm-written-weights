
import numpy as np
from numpy.linalg import inv, slogdet
X=np.load("train_x.npy"); y=np.load("train_y.npy")
def qda_fit(X,y,reg=0.0,pool=0.0):
    K=4; mus=[];As=[];cs=[]
    for k in range(K):
        Xk=X[y==k]; mu=Xk.mean(0); S=np.cov(Xk.T)+reg*np.eye(X.shape[1])
        A=inv(S); mus.append(mu); As.append(A); cs.append(-0.5*slogdet(S)[1]+np.log(len(Xk)/len(y)))
    return mus,As,cs
def qda_pred(m,X):
    mus,As,cs=m
    sc=np.stack([-0.5*np.einsum('ij,jk,ik->i',X-mu,A,X-mu)+c for mu,A,c in zip(mus,As,cs)],1)
    return sc.argmax(1)
rng=np.random.default_rng(0); idx=rng.permutation(len(y)); folds=np.array_split(idx,8)
def parts(m):
    mus,As,cs=m
    R=sum(As)/4
    return [(A-R,A@mu,-0.5*mu@A@mu+c) for mu,A,c in zip(mus,As,cs)]
def trunc_pred(m,X,Xref,r):
    sc=[]
    for D,b,d in parts(m):
        lam,V=np.linalg.eigh(D)
        imp=np.abs(lam)*((Xref@V)**2).mean(0)
        sel=np.argsort(-imp)[:r]
        Dr=(V[:,sel]*lam[sel])@V[:,sel].T
        sc.append(-0.5*np.einsum('ij,jk,ik->i',X,Dr,X)+X@b+d)
    return np.stack(sc,1).argmax(1)
