
import numpy as np
K=21.94; PHI=4.817
def silu(z): return z/(1+np.exp(-np.clip(z,-60,60)))
def build(NL=13,NR=10,w=0.1,G=4.0,S=200.0,SH=2000.0,GH=10.0,G2=20.0,rmax=1.12):
    W1=np.zeros((48,2));b1=np.zeros(48)
    i=0; Lrows=[]
    D=np.pi/NL
    for j in range(NL):
        t=(j+0.5)*D; v=np.array([-np.sin(t),np.cos(t)])
        W1[i]=G/w*v; b1[i]=G; W1[i+1]=G/w*v; b1[i+1]=-G; Lrows.append((i,i+1)); i+=2
    Rrows=[]
    for j in range(NR):
        t=j*np.pi/NR; u=np.array([np.cos(t),np.sin(t)])
        W1[i]=S*u; W1[i+1]=-S*u; Rrows.append((i,i+1)); i+=2
    th=np.linspace(0,np.pi,20001); sm=sum(np.abs(np.cos(th-j*np.pi/NR)) for j in range(NR))
    kap=2/(sm.max()+sm.min())
    hrow=i; W1[i]=[0,-SH*GH]; b1[i]=0.5*GH; W1[i+1]=[0,-SH*GH]; b1[i+1]=-0.5*GH; i+=2
    assert i<=48
    # linear maps from layer1 outputs to L, r, h (each = coef . h1 + const)
    cL=np.zeros(48); cR=np.zeros(48); cH=np.zeros(48)
    for a,b in Lrows: cL[a]+=D/2/G; cL[b]-=D/2/G
    L0=-D/2*NL
    for a,b in Rrows: cR[a]+=kap/S; cR[b]+=kap/S
    cH[hrow]=1/GH; cH[hrow+1]=-1/GH
    W2=np.zeros((48,48)); b2=np.zeros(48); vout=np.zeros(48); n=0
    for half in (0,1):
        sg=1 if half==0 else -1
        # psi = sg*(L+pi/2) + PHI - K r
        cpsi=sg*cL - K*cR; c0=sg*(L0+np.pi/2)+PHI
        lo=PHI-K*rmax-(np.pi if half else 0); hi=PHI+(np.pi if half==0 else 0)
        m0=np.floor((lo-0.3)/np.pi); bs=m0*np.pi
        slope=1.0 if (int(m0)%2==0) else -1.0
        kinks=[(bs,slope)]; kk=bs+np.pi/2; s=slope
        while kk<hi+0.3:
            kinks.append((kk,-2*s)); s=-s; kk+=np.pi
        B=G2*(hi-bs+np.pi)+20
        for bn,vn in kinks:
            W2[n]=G2*cpsi; b2[n]=G2*(c0-bn)
            if half==0: W2[n]+=-B*cH
            else: W2[n]+=B*cH; b2[n]+=-B
            vout[n]=vn/G2; n+=1
    assert n<=48, n
    W3=np.zeros((2,48)); W3[1]=vout/2; W3[0]=-vout/2; b3=np.zeros(2)
    return dict(W1=W1,b1=b1,W2=W2,b2=b2,W3=W3,b3=b3,n2=n)
def fwd(P,X):
    h=silu(X@P['W1'].T+P['b1']); h=silu(h@P['W2'].T+P['b2']); return h@P['W3'].T+P['b3']
def rule(X):
    r=np.hypot(X[:,0],X[:,1]); th=np.arctan2(X[:,1],X[:,0]); return (np.sin(th-K*r+PHI)>0).astype(int)
