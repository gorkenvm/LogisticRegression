import pandas as pd
import numpy as np
df=pd.read_csv(r"C:\Users\DELL\Desktop\python\vscode\Advertising.csv")
df=df.iloc[:,1:len(df)]
#print(df.head())
X=df.drop('sales',axis=1)
y=df[['sales']]
# Statsmodels ile model kurmak
import statsmodels.api as sm
lm=sm.OLS(y,X)
model=lm.fit()
#print(model.summary())
#sklearn ile katsayıları bulma
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
model=lm.fit(X,y)
b0=model.intercept_ # b0 sabiti  
bs=model.coef_  # b değişkenleri
# print(y,bs)   model kurulmuş oldu.

# --------------------------------------
# Tahmin
#  Sales= 2.94 + TV*0.04 + radio*0.19 - newspaper*0.001

#Soru: 30 birim TV, 10 Birim radio, 40 birim gazete harcaması sonucu nedir???

yeni_veri= [[100],[100],[100]]
yeni_veri=pd.DataFrame(yeni_veri).T
model.predict(yeni_veri)
#print(model.predict(yeni_veri))

from sklearn.metrics import mean_squared_error
MSE=mean_squared_error(y,model.predict(X)) # y(gerçek) değeri ve y şapka(tahminedilen) değeridir
RMSE=np.sqrt(MSE)
#print(MSE)
#print(RMSE)

# ---------------------------------------
# Model Tuning (Model Doğrulama)

# Sınama Seti
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=99)
lm=LinearRegression()
lm.fit(X_train,y_train)
np.sqrt(mean_squared_error(y_train, model.predict(X_train))) #Eğitim Hatası
np.sqrt(mean_squared_error(y_test, model.predict(X_test))) # Test Hatası

# k-katlı cv(cross validation) hangi %20 yi test olarak seçecek bunun sonucunun en iyi çıkmasını sağlayacak %20 yi seçer

from sklearn.model_selection import cross_val_score
cv=cross_val_score(model, X_train, y_train, cv=10, scoring="neg_mean_squared_error")
cvmse=np.mean(-cross_val_score(model, X_train, y_train, cv=10, scoring="neg_mean_squared_error")) # Cross Valide edilmiş MSE
cvrmse=np.sqrt(np.mean(-cross_val_score(model, X_train, y_train, cv=10, scoring="neg_mean_squared_error"))) # Cross Valide edilmiş RMSE
#print(cv,cvmse,cvrmse)

















