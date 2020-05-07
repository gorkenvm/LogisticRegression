# RİDGE MODELS

### Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import model_selection

#-------------------------------------------------------------
# Data Set
df=pd.read_csv(r"C:\Users\DELL\Desktop\python\vscode\hitter.csv")
df=df.dropna()
dms=pd.get_dummies(df[['League', 'Division', 'NewLeague']])
y=df["Salary"]
X_=df.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')
X=pd.concat([X_, dms[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)
ridge_model= Ridge(alpha=0.1).fit(X_train, y_train)
ridge_model.intercept_
lambdalar=10**np.linspace(10,-2,100)*0.5
#print(lambdalar)
ridge_model=Ridge()
katsayilar=[]

for i in lambdalar:
    ridge_model.set_params(alpha=i)
    ridge_model.fit(X_train, y_train)
    katsayilar.append(ridge_model.coef_)
#print(katsayilar)
ax=plt.gca()
ax.plot(lambdalar, katsayilar)
ax.set_xscale("log")
#plt.show(ax)

#--------------------------------------------------------------
### GUESS

ridge_model=Ridge().fit(X_train,y_train) # Model kurma işlemi
y_pred=ridge_model.predict(X_train) # y değerlerini tahmin etme
RMSE=np.sqrt(mean_squared_error(y_train,y_pred)) #train hatası
np.sqrt(np.mean(-cross_val_score(ridge_model,X_train, y_train, cv=10 , scoring="neg_mean_squared_error"))) #crossvalidation yapıldı
y_pred=ridge_model.predict(X_test) #test hatası
RMSE=np.sqrt(mean_squared_error(y_test,y_pred))
#print(RMSE)
#------------------------------------------------------
## MODEL TUNNİNG
###bağımlı ve bağımsız veriler arasındaki ilişkidir Modellemek 
####öğrendiğimiz veriyi bir fonksiyon ile kullanmaktır tahmin etmek

ridge_model=Ridge().fit(X_train, y_train) # train veri setini kullanarak model oluşturduk.
y_pred=ridge_model.predict(X_test) # kurduğumuz model ile X_test verilerinden y verilerini tahmin ettik
rmse=np.sqrt(mean_squared_error(y_test, y_pred)) # gerçek y ve tahmin edilen y değerleri ile hata ortalamarı karasinin kökünü bulduk
lambdalar1=np.random.randint(0,1000,100) #ridge_model içerisinde bulunan Ridge() lamda değerini girmemize yarar. fakat lambda değerini bilmediğimiz için 0-1000 arası 100 tane sayı rastgele oluşturuyoruz.
lambdalar2=10**np.linspace(10,-2,100)*0.5 #buda lambdalar1 gibi üretilmiş değerler. bunu yada lambda1 i kullanabilirsin. ikisini kullanıp MSE yi karşılaştırabilirsin
ridgecv=RidgeCV(alphas=lambdalar2, scoring="neg_mean_squared_error", normalize=True) # lambdalar2 yi kullanarak alpha'lar yani lamdalar değerleri bulacak
ridgecv.fit(X_train,y_train) #fit ettik
#print(ridgecv)
ridgecv.alpha_ # ridgecvden gelen optimum alphayı alıyor.

#Final Model
ridge_tuned=Ridge(alpha=ridgecv.alpha_).fit(X_train, y_train) # tuned edilmiş ridge 
y_pred=ridge_tuned.predict(X_test) # ridge_tuned modelini kularak X_test verileri ile y değerlerini tahmin et
np.sqrt(mean_squared_error(y_test, y_pred)) # MSE











