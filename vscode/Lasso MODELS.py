# Lasso MODELS

### Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
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

# SET MODEL
lasso_model=Lasso().fit(X_train, y_train) # Model kuruldu
lasso_model.intercept_ # sabiti aldık
lasso_model.coef_      # katsayıları aldık

#farklı lambda degerlerine karşılık katsayılar
lasso=Lasso()
coefs=[]      #katsayılar listesi
#alphas=np.random.randint(0,1000,10) # sayı türetiyoruz ki en iyi alphayı bulalım
alphas=10**np.linspace(10,-2,100)*0.5 # birde bunu deneyelim. çok daha iyi sonuç veriyormuş.
for a in alphas:
    lasso.set_params(alpha=a) #a ları alphas(türetilen rakamlarda )gezdir
    lasso.fit(X_train, y_train) # fit et
    coefs.append(lasso.coef_) # katsayıları coefs listesine ekle

#görselleştirelim
ax=plt.gca()
ax.plot(alphas,coefs)
ax.set_xscale("log")
#plt.show(ax)

# GUESS

#lasso_model.predict(X_train) # modelde X_traini kullanarak tahmin et
y_pred=lasso_model.predict(X_test)  # modelde X_test kullanarak tahmin et
mse=np.sqrt(mean_squared_error(y_test, y_pred))
r2_score(y_test, y_pred) #bağımsız değişkenlerce bağımlı değişkendeki değişimin, değişme yüzdesidir 



#MODEL TUNNİNG

lasso_cv_model= LassoCV(cv=10, max_iter=100000).fit(X_train, y_train) #cross validation yapıldı
lasso_cv_model.alpha_ # alphayı çek
lasso_tuned=Lasso(alpha=lasso_cv_model.alpha_).fit(X_train, y_train) # model tunned edildi
y_pred=lasso_tuned.predict(X_test)
mse=np.sqrt(mean_squared_error(y_test, y_pred))
#print(mse)

pd=pd.Series(lasso_tuned.coef_, index=X_train.columns)
#print(pd)




