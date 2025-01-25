import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import NearMiss
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("Creditcard_data.csv")

# print(df.columns)
df_majority=df[df['Class']==0]
df_minority=df[df['Class']==1]

df_minority_unsampled=resample(df_minority,
                               replace=True,
                               n_samples=len(df_majority),
                               random_state=42)
df_bal=pd.concat([df_majority,df_minority_unsampled])
print(f"balanced dataset shape is  {df_bal.shape}")

sample_size=100
samples=[df_bal.sample(n=sample_size,random_state=i) for i in range(5)]

sampling_res={}

models={
    'm1':RandomForestClassifier(random_state=42),
    'm2':LogisticRegression(random_state=42 , max_iter=1000),
    'm3':SVC(random_state=42),
    'm4':DecisionTreeClassifier(random_state=42),
    'm5':KNeighborsClassifier()
}
def sampling_tech(df):
    df_minority_unsampled=resample(
        df[df['Class']==1],
        replace=True,
        n_samples=len(df[df['Class']==0]),
        random_state=42
    )
    df_sample1=pd.concat([df[df['Calss']==0],df_minority_unsampled])


    df_minority_undersample=resample(
        df[df['Class']==1],
        replace=False,
        n_samples=len(df[df['class']==0]),
        random_state=42
    )
    df_sample2=pd.concat([df[df['Class']==0],df_minority_undersample])

    nearmiss=NearMiss()
    x=df.drop(columns=['Class'])
    y=df['Class']
    x_res,y_res=nearmiss.fit_resample(x,y)
    df_sample3=pd.concat([pd.DataFrame(x_res,columns=X.columns),pd.DataFrame(y_res,columns=['Class'])],axis=1)

    return [df_sample1,df_sample2,df_sample3]
scaler=StandardScaler()
for i,sample in enumerate(samples):
    x=sample.drop(columns=['Class'])
    y=sample['Class']
    x_train,x_test,y_train,y_test=train_test_split(
        x,y,test_size=0.3,random_state=42
    )

    x_train_scaled=scaler.fit_transform(x_train)
    x_test_scaled=scaler.transform(x_test)
    model_res={}
    for model_name,mod in models.items():
        mod.fit(x_train_scaled,y_train)
        y_pred=mod.predict(x_test)
        accuracy=accuracy_score(y_test,y_pred)
        model_res[model_name]=accuracy
    sampling_res[f"sampling{i+1}"]=model_res
    
for sampling,res in sampling_res.items():
    print(f"\n{sampling} results")
    for mod_name,accuracy in res.items():
        print(f"{model_name}: {accuracy:.2f}")