import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# constants 
Z=1.96
p=0.5
E=0.05
S=5
C=10


df=pd.read_csv("Creditcard_data.csv")

# print(df.columns)
df_majority=df[df['Class']==0]
df_minority=df[df['Class']==1]

x=df.drop(columns=['Class'],axis=1)
y=df['Class']

smote=SMOTE(random_state=42)
x_bal,y_bal=smote.fit_resample(x,y)

models={
    'm1':RandomForestClassifier(random_state=42),
    'm2':LogisticRegression(random_state=42 , max_iter=2000),
    'm3':SVC(random_state=42),
    'm4':DecisionTreeClassifier(random_state=42),
    'm5':KNeighborsClassifier()
}
def simple_random_sample(x_bal,y_bal,n):
    indices=np.random.choice(x_bal.index,size=n,replace=False)
    return x_bal.loc[indices],y_bal.loc[indices]

def stratified_sample(x_bal,y_bal,strata_count):
    strata=pd.qcut(y_bal.rank(method='first'),q=strata_count,labels=False)
    sam_x,sam_y=[],[]
    for grp in range(strata_count):
        grp_idx=strata[strata==grp].index
        sam_ind=np.random.choice(grp_idx,size=len(grp_idx))
        sam_x.append(x_bal.loc[sam_ind])
        sam_y.append(y_bal.loc[sam_ind])
    return pd.concat(sam_x),pd.concat(sam_y)

def cluster_sample(x_bal,y_bal,n_clusters):
    kmeans=KMeans(n_clusters=n_clusters,random_state=42)
    x_bal['cluster']=kmeans.fit_predict(x_bal)
    sam_x = pd.concat([x.sample(1, random_state=42) for _, x in x_bal.groupby('cluster')]).reset_index(drop=True)
    sam_x = sam_x[x_bal.columns.difference(['cluster'])]  
    sam_y=y_bal.loc[sam_x.index]
    return sam_x,sam_y

n_simple=1000
n_stratified=5
n_clusters=5

samples = {
    'Simple Random': simple_random_sample(x_bal, y_bal, n_simple),
    'Stratified': stratified_sample(x_bal, y_bal, n_stratified),
    'Cluster': cluster_sample(x_bal, y_bal, n_clusters)
}

res=[]
scaler=StandardScaler()
for sample_name, (x_sam,y_sam) in samples.items():
    x_train,x_test,y_train,y_test=train_test_split(
    x_sam,y_sam,test_size=0.2,random_state=42)
    if len(np.unique(y_train))>1:
        scaler=StandardScaler()
        x_train=scaler.fit_transform(x_train)
        x_test=scaler.transform(x_test)
        for mod_name,mod in models.items():
            if isinstance(mod, KNeighborsClassifier) and len(x_sam) < 5:
                mod = KNeighborsClassifier(n_neighbors=len(x_sam)) 
            mod.fit(x_train,y_train)
            y_pred=mod.predict(x_test)
            accuracy=accuracy_score(y_test,y_pred)
            res.append({'Model': mod_name, 'Sampling Technique': sample_name, 'Accuracy': accuracy})
    else:
        print(f"skipping model training for {sample_name} as it contains only onr class in training set")

results_df = pd.DataFrame(res)
pivot_table = results_df.pivot(index='Model', columns='Sampling Technique', values='Accuracy')
pivot_table.to_csv('model_sampling_results.csv')
print(pivot_table)