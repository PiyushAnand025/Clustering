# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,0:2].values
a=np.shape(X)
ns=a[0]                              #number of records         
nd=a[1]                              #number of features
cluster=[]                           # A list to include all clusters
Update=[]                           # A list to include all partion martix  
IKK=[]
XBB=[]
Dunnn=[]      
A=np.zeros([nd,nd],dtype=float)
B=np.zeros([nd,nd],dtype=float)
# Calculatin A norm
icon=int(input("Press 1 for Euclidean Norm, 2 for Diagonal Norm, 3 for Mahalonobis Norm\n"))
if (icon==1):
    for i in range(nd):
        for j in range(nd):
            if(i==j):
                A[i][j]=1
            else:
                A[i][j]=0
elif (icon==2):
    cy=np.sum(X,axis=0,keepdims=True)
    cy=cy/ns
    Cy=np.zeros([nd,nd],dtype=float)
    for i2 in range(ns):
        b=np.dot(np.transpose(X[i2,:]-cy),X[i2,:]-cy)
        Cy=np.add(Cy,b)
        
    W=np.linalg.eigvals(Cy)
    
    for i in range(nd):
        for j in range(nd):
            if(i==j):
                B[i][j]=W[i]
            else:
                B[i][j]=0
                
    A=np.linalg.inv(B)
    
else:
    cy=np.sum(X,axis=0,keepdims=True)
    cy=cy/ns
    Cy=np.zeros([nd,nd],dtype=float)
    for i2 in range(ns):
        b=np.dot(np.transpose(X[i2,:]-cy),X[i2,:]-cy)
        Cy=np.add(Cy,b)
        
    A=np.linalg.inv(Cy)
    
n=np.shape(X)[0]
k=np.shape(X)[0]
a=k**0.5
a=int(a)
    # Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, a):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, a), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
    
    
m=float(input("Enter The value of weight exponent m between 1.5 and 3"))
kbegin=2
kcease=int(input("Enter the total number of cluster"))
eps=0.01            #Ending Criteria
lmax=50             #Ending Criteria
jm=np.zeros([kcease-1],dtype=float)        #Cost Function
fc=np.zeros([kcease-1],dtype=float)         #Partition Coefficient
hc=np.zeros([kcease-1],dtype=float)        #Entropy
dif=np.zeros([kcease-1],dtype=float)       #1-fc

for zz in range(2,kcease+1):
    c=zz                         #Number of cluster
    print(c)
    U=np.random.rand(c,ns)        # Fuzzy matrix
    b=np.sum(U,axis=0,keepdims=True)     
    U=U/b                               #adjusting matrices to satisfy equation 2 
    v=np.zeros([c,nd],dtype=float)      #Initial centres
    
    for i in range(c):                  #Calculating Initial clusters
        aa=0
        for j in range(ns):
            aa+=(U[i][j]**m)
            v[i]=v[i]+((U[i][j]**m)*X[j])
        v[i]=v[i]/aa
        
    d=np.zeros([c,ns],dtype=float)         #calculating initial distance matrix
    for i in range(c):
        p=(X-v[i])
        q=np.dot(p,A)
        ll=np.dot(q,np.transpose(p))
        for j in range(ns):
            d[i][j]=ll[j][j]
            
            
            
    for mm in range(lmax):                              #calculating clusters
         U2=np.zeros([c,ns],dtype=float)          #Calculating Updated U matrix
         for i in range(c):
             for k in range(ns):
                 aa=0
                 for j in range(c):
                     aa+=((d[i][k]/d[j][k])**(1/(m-1)))
                     
                 U2[i][k]=1/aa
                    
         if(np.max(U2-U)<=eps):                   #checking condition
            break
         else:
             U=U2
             for i in range(c):
                 aa=0
                 v[i]=0                             #Updating v matrix
                 for j in range(ns):
                     aa+=(U[i][j]**m)
                     v[i]=v[i]+((U[i][j]**m)*X[j])
                 v[i]=v[i]/aa
                
                
             for i in range(c):                      #Updating distance matrix
                 p=(X-v[i])
                 q=np.dot(p,A)
                 ll=np.dot(q,np.transpose(p))
                 for j in range(ns):
                    d[i][j]=ll[j][j]
                    
    Update.append(U2)                
    cluster.append(v)                              # Storing Clusters           
    for k in range(c):
        for i in range(ns):
            jm[zz-2]=jm[zz-2]+(U[k][i]**m)*d[k][i]
            fc[zz-2]=fc[zz-2]+((U[k][i]**2)/ns)
            hc[zz-2]=hc[zz-2]+(((U[k][i]*np.log(U[k][i]))/ns)*(-1))
            
        dif[zz-2]=1-fc[zz-2]
    U=U2                        
    cd=np.zeros([c,c],dtype=float)      #getting cluster distance matrix
    for i in range(c):
        for j in range(c):
            for k in range(nd):
                if v[i][k]!=v[j][k]:
                    cd[i][j]+=1
                    
    #Calculating XB Index
    d2=np.multiply(d,d)
    U2=np.multiply(U,U)
    var1=np.sum(np.multiply(U2,d2))
    
    minsep=nd                
    for i in range(c):
       for j in range(c):
         if cd[i][j]>0 and cd[i][j]<minsep:
              minsep=cd[i][j]
                
    XB=var1/ns
    XB=XB/minsep
    print("%%%%%%%%%XB  {0}".format(XB))

    # Calculating Dunn Index
    delta=np.multiply(U,d)
    delta2=np.sum(delta,axis=1)
    Xk=np.max(delta2)
    dunn=np.inf
    maxdunn=0
    maxc=0
    for i in range(c-1):
        for j in range(i+1,c):
            gg=cd[i][j]/Xk
            if gg<dunn:
                dunn=gg
            if dunn>maxdunn:
                maxdunn=dunn
                maxc=c
    
            
    
    print("*********Dunn  {0}".format(dunn))
    
    #calculating I Index
    Dk=np.max(cd)
    Ek=np.sum(np.multiply(U,d))
    E1=np.sum(np.multiply(U,d),axis=1)[0]
    Ik=(Dk*E1)/(Ek*c)
    
    print("##########IK   {0}".format(Ik))
    IKK.append(Ik)
    XBB.append(XB)
    Dunnn.append(dunn)        
       
XBB=np.array(XBB)
Dunnn=np.array(Dunnn)
IKK=np.array(IKK)

#gh=np.array(range(2,kcease+1))
#
#
#plt.scatter(gh,XBB,c='r')
#plt.scatter(gh,Dunnn,c='b')
#plt.scatter(gh,IKK,c='y')


#Applying SVM to enhance clustering
oc=int(input("Enter the optimal number of cluster"))
U3=Update[oc-2]
U3=np.array(U3)        
e=[]
mar=float(input("Enter the marginal value"))
for i in range(oc):
    li=[]
    for j in range(ns):
        if(U3[i][j]>=mar):
            #print(j)
            li.append(j)
    e.append(li)

Xtrain=[]
ytrain=[]
nshape=[]
count=0        
for i in e:
    for k in i:
        nshape.append(k)
        Xtrain.append(X[k])
        ytrain.append(count)
    count+=1

Xtrain=np.array(Xtrain)
ytrain=np.array(ytrain)

nshape=np.array(nshape)

Xtest=[]
for j in range(ns):
    if j not in nshape:
        Xtest.append(X[j])
        

Xtest=np.array(Xtest) 

# Fitting Kernel SVM to the Training set
#from sklearn.svm import SVC
#classifier = SVC(kernel = 'rbf', random_state = 0)
#classifier.fit(Xtrain, ytrain)           
import xgboost as xgb
model=xgb.XGBClassifier(random_state=42,learning_rate=0.01)
model.fit(Xtrain, ytrain)
y_pred = model.predict(Xtest)
ypred = model.predict(X)

# Predicting the Test set results
#y_pred = classifier.predict(Xtest)
#
#ypred = classifier.predict(X)
        


    
cl=cluster[oc-2]
print(cl)
d=0
for i in range(oc):
    for j in range(ns):
        if(ypred[j]==i):
            d=d+(X[j][0]-cl[i][0])**2+(X[j][1]-cl[i][1])**2
    
    
    
    
    
    
    
    
    
    
    