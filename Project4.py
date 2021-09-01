# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data2.csv')
X = dataset.iloc[:50,0:5].values
X2=pd.DataFrame(X)
a=np.shape(X)
ns=a[0]                              #number of records         
nd=a[1]                              #number of features

D=np.zeros([ns,ns],dtype=float)      #calculating distance matrix
for i in range(ns):
    for j in range(ns):
        sum=0
        for k in range(nd):
            if X[i][k]!=X[j][k]:
                sum+=1
        D[i][j]=sum
        
        
p=int(input("Enter a population size"))
c=int(input("Enter the total number of clusters"))
m=float(input("Enter The value of weight exponent m between 1.5 and 3"))

cluster=[]
clus=[]                           #initializing population
for i in range(p):
    cluster.append(np.random.randint(0,ns,size=c))
    
cluster=np.array(cluster)

for i in cluster:
    v=i
    d=[]
    for j in i:
        d.append(D[j])
    d=np.array(d)
    
    U=np.zeros([c,ns],dtype=float)      #getting initial fuzzy matrix 
    v=np.array(v)
    V=[]
    for q in v:
        V.append(X[q])
    V=np.array(V)
    ci=[]
    xi=[]
    for j in range(ns):
        for x in range(c):
            cou=0
            for y in range(nd):
                if X[j][y]==V[x][y]:
                    cou+=1
                
                if cou==nd:
                    ci.append(x)
                    xi.append(j)
    ns2=[]            
    for k in range(ns):
        if k not in xi:
            ns2.append(k)
                
                
    for x in range(len(ci)):
            U[ci[x]][xi[x]]=1
            
            
        
    for ii in ns2:
        for jj in range(c):
            su=0
            for kk in range (c):
                su=su+((d[jj][ii]/d[kk][ii])**(1/(m-1)))
                    
            U[jj][ii]=(1/su)
            
    for pp in range(c):          #Updating the cluster to get initial cluster.
        su=[]
        for qq in range(ns):
            sum=0
            for rr in range(ns):
                sum=sum+((U[pp][rr])**m)*D[qq][rr]
            su.append(sum)
        
        #print(su)    
        clus.append(su.index(min(su)))
        
clus=np.array(clus)
clus=np.reshape(clus,(p,c))

jm=[]
for i in clus:                #calculating fitness
    v=i
    d=[]
    for j in v:
        d.append(D[j])
    d=np.array(d)
    
    U=np.zeros([c,ns],dtype=float)      #getting  fuzzy matrix 
    v=np.array(v)
    V=[]
    for q in v:
        V.append(X[q])
    V=np.array(V)
    ci=[]
    xi=[]
    for j in range(ns):
        for x in range(c):
            cou=0
            for y in range(nd):
                if X[j][y]==V[x][y]:
                    cou+=1
                
                if cou==nd:
                    ci.append(x)
                    xi.append(j)
    ns2=[]            
    for k in range(ns):
        if k not in xi:
            ns2.append(k)
                
                
    for x in range(len(ci)):
            U[ci[x]][xi[x]]=1
            
            
        
    for ii in ns2:
        for jj in range(c):
            su=0
            for kk in range (c):
                su=su+((d[jj][ii]/d[kk][ii])**(1/(m-1)))
                    
            U[jj][ii]=(1/su)
            
    sum=0
    for kk in range(c):
        for jj in range(ns):
            sum=sum+(U[kk][jj]**m)*d[kk][jj]
        
    jm.append(sum)

jm=np.array(jm)
jm=np.reshape(jm,(p,1))

Gb=np.amin(jm)
lb=Gb

x=np.where(jm==np.amin(jm))
x=np.array(x)
Gbest=clus[x[0]]
Gbest=Gbest[0]
Glbest=Gbest

cr=float(input("Enter the crossover value between 0 and 1"))
lmax=int(input("Enter the total number of iterations"))
f=int(input("Enter the value of F"))
for zz in range(lmax):
    al=1/(1+np.exp(-zz))
    cl=[]
    jl=[]
    for iii in range(p):
        i=clus[iii]
        
        if(np.random.uniform()<al):         #Mutation
            vi=Gbest+(Glbest-i)*f
        else:
            ab=np.random.randint(0,p,size=2)
            vi=clus[ab[0]]+(clus[ab[1]]-i)*f
    
        for k in range(c):                       #adjustment
            while(vi[k]>=ns or vi[k]<0):
                if(vi[k]>=ns):
                    vi[k]=vi[k]-ns
                if(vi[k]<0):
                    vi[k]=vi[k]+ns
                    
        UU=[]
        pp=np.random.randint(0,c,size=1)
        for k in range(c):
            ll=np.random.uniform()
            if(ll<=cr or k==pp):
                UU.append(vi[k])
            else:
                UU.append(i[k])
                
        UU=np.array(UU)
        #print(UU)
        
    
        v=UU
        d=[]
        for j in v:       #making distance matrix
            d.append(D[j])
        d=np.array(d)
    
        U=np.zeros([c,ns],dtype=float)      #getting  fuzzy matrix 
        v=np.array(v)
        V=[]
        for q in v:                       #getting values from dataset
            V.append(X[q])
        V=np.array(V)
        ci=[]
        xi=[]
        for j in range(ns):
            for x in range(c):
                cou=0
                for y in range(nd):
                    if X[j][y]==V[x][y]:
                        cou+=1
                
                if cou==nd:
                    ci.append(x)
                    xi.append(j)
        ns2=[]            
        for k in range(ns):
            if k not in xi:
                ns2.append(k)
                
                
        for xx in range(len(ci)):
            U[ci[xx]][xi[xx]]=1
            
            
        
        for ii in ns2:
            for jj in range(c):
                su=0
                for kk in range (c):
                    su=su+((d[jj][ii]/d[kk][ii])**(1/(m-1)))
                    
                U[jj][ii]=(1/su)
            
        sum=0                            #calculating fitness
        for kk in range(c):
            for jj in range(ns):
                sum=sum+(U[kk][jj]**m)*d[kk][jj]
        #print(iii)
        if(sum<=jm[iii]):
            jl.append(sum)
            cl.append(UU)
        else:
            jl.append(jm[iii])
            cl.append(i)
            
        
    jl=np.array(jl)
    jl=np.reshape(jl,(p,1))
    jm=jl
    cl=np.array(cl)
    x=np.where(jl==np.amin(jl))
    x=np.array(x)
    Glbest=cl[x[0]] 
    Glbest=Glbest[0]
    lb=np.amin(jl)
    if(lb<Gb):
        Gb=lb
        Gbest=Glbest
        
    clus=cl
    
print(Gb)
print(Gbest)
    
v=Gbest
d=[]
for j in v:       #making distance matrix
    d.append(D[j])
d=np.array(d)
    
U=np.zeros([c,ns],dtype=float)      #getting  fuzzy matrix 
v=np.array(v)
V=[]
for q in v:                       #getting values from dataset
    V.append(X[q])
V=np.array(V)
ci=[]
xi=[]
for j in range(ns):
    for x in range(c):
        cou=0
        for y in range(nd):
            if X[j][y]==V[x][y]:
                cou+=1
                
        if cou==nd:
            ci.append(x)
            xi.append(j)
ns2=[]            
for k in range(ns):
    if k not in xi:
        ns2.append(k)
                
                
for xx in range(len(ci)):
    U[ci[xx]][xi[xx]]=1
            
            
        
for ii in ns2:
    for jj in range(c):
        su=0
        for kk in range (c):
            su=su+((d[jj][ii]/d[kk][ii])**(1/(m-1)))
                    
        U[jj][ii]=(1/su)
            
sum=0                            #calculating fitness
for kk in range(c):
    for jj in range(ns):
        sum=sum+(U[kk][jj]**m)*d[kk][jj]


print("Clusters are")        
for i in range(len(Gbest)):
    print(X[Gbest[i]])
    
    
    
#Applying SVM to enhance clustering
        
#e=[]
#mar=float(input("Enter the marginal value"))
#for i in range(c):
#    li=[]
#    for j in range(ns):
#        if(U[i][j]>=mar):
#            #print(j)
#            li.append(j)
#    e.append(li)
#
#Xtrain=[]
#ytrain=[]
#nshape=[]
#count=0        
#for i in e:
#    for k in i:
#        nshape.append(k)
#        Xtrain.append(X[k])
#        ytrain.append(count)
#    count+=1
#
#Xtrain=np.array(Xtrain)
#ytrain=np.array(ytrain)
#
#nshape=np.array(nshape)
#
#Xtest=[]
#for j in range(ns):
#    if j not in nshape:
#        Xtest.append(X[j])
#        
#
#Xtest=np.array(Xtest) 
#
## Fitting Kernel SVM to the Training set
#from sklearn.svm import SVC
#classifier = SVC(kernel = 'rbf', random_state = 0)
#classifier.fit(Xtrain, ytrain)           
#
## Predicting the Test set results
#y_pred = classifier.predict(Xtest)
#
#ypred = classifier.predict(X)
        

        
        

    
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
