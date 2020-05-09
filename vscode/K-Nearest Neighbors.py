#K-En Yakın Komşu 
# Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import neighbors
from sklearn.svm import SVR
# ------------------------------------------
#Hata Silme
from warnings import filterwarnings
filterwarnings('ignore')
# -----------------------------------------------------------------------------

# KNN
# Read data
df=pd.read_csv(r"C:\Users\DELL\Desktop\python\vscode\LinearRegressionModels\hitter.csv")
df=df.dropna()
dms=pd.get_dummies(df[['League','Division', 'NewLeague']])
y=df["Salary"]
X_=df.drop(['Salary','League','Division','NewLeague'], axis=1).astype('float64')
X=pd.concat([X_,dms[['League_N','Division_W','NewLeague_N']]],axis=1)
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.25, random_state=42)
#print(X_train.head())

# Model Oluşturma & Tahmin Etme
knn_model=KNeighborsRegressor().fit(X_train,y_train) #model kuruldu
y_pred=knn_model.predict(X_test) # y tahmin edildi
#print(np.sqrt(mean_squared_error(y_test,y_pred)))


# Model Tunning

RMSE=[]
for k in range(10):
    k =k+1
    knn_model=KNeighborsRegressor(n_neighbors=k).fit(X_train,y_train)
    y_pred=knn_model.predict(X_test)
    rmse=np.sqrt(mean_squared_error(y_test,y_pred))
    RMSE.append(rmse)
    #print("k=",k,"için RMSE değeri:",rmse)

#GridSearchCV Yukarıda for ile yaptığımız işi otomatik yapar
knn_params={"n_neighbors": np.arange(1,30,1)} #k parametreleri 1 den 30 a kadar al
knn=KNeighborsRegressor() #Modelledik
knn_cv_model=GridSearchCV(knn,knn_params, cv=10).fit(X_train, y_train) # modelimizi yani knn, parametreleri yani knn_params, cv yi yazdık. CV model kuruldu.
#print(knn_cv_model.best_params_) # en iyi k deperini buluyor 8 buldu.

#Final Model
knn_tuned=KNeighborsRegressor(n_neighbors=knn_cv_model.best_params_["n_neighbors"]).fit(X_train, y_train)
y_pred=knn_tuned.predict(X_test)
rmset=np.sqrt(mean_squared_error(y_test, y_pred))
#print(rmset)













