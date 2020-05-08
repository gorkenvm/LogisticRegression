# ElasticNet Regression Model
## Make İt Togetjer of Ridge and Lasso 

### Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import model_selection

#-------------------------------------------------------------
# DATA SET
df=pd.read_csv(r"C:\Users\DELL\Desktop\python\vscode\hitter.csv")
df=df.dropna()
dms=pd.get_dummies(df[['League', 'Division', 'NewLeague']])
y=df["Salary"]
X_=df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')
X=pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)



enet_model=ElasticNet().fit(X_train, y_train) #model kuruldu
enet_model.coef_ #sabit terim
enet_model.intercept_ #kat sayılar
enet_model.predict(X_train) #tahmin
y_pred=enet_model.predict(X_test)
MSE=np.sqrt(mean_squared_error(y_test, y_pred))
#print(MSE)
r2_score(y_test, y_pred) # isteğe bağlı


#Model Tunning

enet_cv_model=ElasticNetCV(cv=10).fit(X_train, y_train)
enet_cv_model.alpha_
enet_cv_model.intercept_
enet_cv_model.coef_

#Final Model
enet_tuned=ElasticNet(alpha=enet_cv_model.alpha_).fit(X_train, y_train) # tuned edilmiş modeli kurup fit ettik
y_pred=enet_tuned.predict(X_test)
MSEt=np.sqrt(mean_squared_error(y_test, y_pred))
#print("Tuned MSE:",MSEt)




