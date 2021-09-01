# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data2.csv')
X = dataset.iloc[:,0:5].values
a=np.shape(X)
ns=a[0]                              #number of records         
nd=a[1]                              #number of features
cluster=[]                           # A list to include all clusters
Update=[]                           # A list to include all partion martix 
m=float(input("Enter The value of weight exponent m between 1.5 and 3"))
kbegin=2
kcease=int(input("Enter the total number of cluster"))
jm=np.zeros([kcease-1],dtype=float)        #Cost Function and ending criterian
lmax=50             #Ending Criteria

for zz in range(2,kcease+1):
    c=zz
    v=[]                           #getting initial clusters
    for i in range(c):
        v.append(X[i])
        
        
       
    d=np.zeros([c,ns],dtype=float)      #getting initial distance matrix
    for i in range(c):
        for j in range(ns):
            for k in range(nd):
                if v[i][k]!=X[j][k]:
                    d[i][j]+=1
                    
    
    
    U=np.zeros([c,ns],dtype=float)      #getting initial fuzzy matrix                
    if m==1:
        cmin=np.amin(d,axis=0)
        for i in range(ns):
            for j in range(c):
                if d[j][i]==cmin[i]:
                    U[j][i]=1
                    break
                else:
                    U[j][i]=0
    else:
        v=np.array(v)
        ci=[]
        xi=[]
        for j in range(ns):
            for x in range(c):
                cou=0
                for y in range(nd):
                    if X[j][y]==v[x][y]:
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
                       
    
        
    Update.append(U)              
    cluster.append(v)         
#    sum1=np.sum(np.multiply(U,d))
    v=np.array(v)

    for mm in range(lmax):
        if m==1:
            #Update.append(U)                
            #cluster.append(v) 
            
            fco=[]
            for i in range(c):
                co=[]
                for j in range(ns):
                    if U[i][j]==1:
                        co.append(j)
                fco.append(co)
            
            k=0
            for cou in fco:
                nX=[]
                for i in cou:
                    nX.append(X[i])
                    
                nX=np.array(nX)
                for i in range(nd):
                    ftable=pd.crosstab(index=nX[:,i],columns='count')
                    amax=ftable['count'].idxmax(axis=0)
                    v[k][i]=amax                #Updating clusters
                k+=1
            
            
            
            d=np.zeros([c,ns],dtype=float)      #Updating distance matrix
            for i in range(c):
                for j in range(ns):
                    for k in range(nd):
                        if v[i][k]!=X[j][k]:
                            d[i][j]+=1        
            
#            sum2=np.sum(np.multiply(U,d))
#            if sum2 >= sum1:
#                break
#            else:
#                sum1=sum2
            
            
            U=np.zeros([c,ns],dtype=float)      #setting initial fuzzy matrix 
            cmin=np.amin(d,axis=0)             #Updating fuzzy matrix
            for i in range(ns):
                for j in range(c):
                    if d[j][i]==cmin[i]:
                        U[j][i]=1
                        break
                    else:
                        U[j][i]=0
                        
#            sum2=np.sum(np.multiply(U,d))
#            if sum2 >= sum1:
#                break
#            else:
#                sum1=sum2
            Update.append(U)                
            cluster.append(v) 
            
            
            
        else:
            #Update.append(U)                
            #cluster.append(v) 
            X2=pd.DataFrame(X)
            for i in range(nd):
                Uc=X2.iloc[:,i].unique().tolist()
                for l in range(c):
                    sm=np.zeros(len(Uc),dtype=float)
                    for j in range(len(Uc)):
                        for k in range(ns):
                            if Uc[j]==X2[i][k]:
                                sm[j]=sm[j]+U[l][k]**m
                    lock=np.where(sm==np.amax(sm))
                    lock=np.array(lock)
                    v[l][i]=Uc[lock[0][0]]    #Updating V matrix
            
            
            
            d=np.zeros([c,ns],dtype=float)        #Updating distance matrix
            for i in range(c):
                for j in range(ns):
                    for k in range(nd):
                        if v[i][k]!=X[j][k]:
                            d[i][j]+=1       
            
            
#            sum2=np.sum(np.multiply(U,d))
#            if sum2 >= sum1:
#                break
#            else:
#                sum1=sum2
            
                
                
            U=np.zeros([c,ns],dtype=float)      #Updating  fuzzy matrix           
            X=np.array(X)
            v=np.array(v)
            ci=[]
            xi=[]
            for j in range(ns):
                for x in range(c):
                    cou=0
                    for y in range(nd):
                        if X[j][y]==v[x][y]:
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
                    
                    
                    
#            sum2=np.sum(np.multiply(U,d))
#            if sum2 >= sum1:
#                break
#            else:
#                sum1=sum2
#            
            Update.append(U)                
            cluster.append(v) 
                    
#print(cluster)                
#print(Update)
    
print(cluster)
print(X)












