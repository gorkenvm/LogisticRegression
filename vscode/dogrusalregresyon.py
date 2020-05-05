import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv(r"C:\Users\DELL\Desktop\python\vscode\Advertising.csv")
df=df.iloc[:,1:len(df)]
###print(df.head())
###print(df.info())
import seaborn as sns
###a=sns.jointplot(x="TV", y="sales", data=df, kind="reg")
###b=sns.jointplot(x="radio", y="sales", data=df, kind="reg")
###plt.show(a),plt.show(a)
from sklearn.linear_model import LinearRegression
X=df[["TV"]]
y=df[["sales"]]
reg=LinearRegression()
model=reg.fit(X,y)
###print(model)
###print(dir(model))
b0=model.intercept_
b1=model.coef_
rkare=model.score(X,y) #rkare,modelin skorudur
### ----------------------------------------------------------------------------------------------
###  TAHMİN ###
g=sns.regplot(df["TV"],df["sales"],ci=None,scatter_kws={'color':'g','s':9})
g.set_title("Model Denklemi: Sales : 7.03 + TV*0.05")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV harcamaları")
plt.xlim(-10,310)
plt.ylim(bottom=0);
###plt.show(g)
###print(model.predict([[165]]))
###-------------------------------------------------------------------------------------------------------------
### Artıklar ve Makine Öğrenmesindeki Önemi
###### MSE: Hata Kareler Ortalaması
###### RMSE: Hata Karaler Ortalamasının Karekökü
gercek_y=y[0:10]
tahminedilen_y=pd.DataFrame(model.predict(X)[0:10])
hatalar=pd.concat([gercek_y,tahminedilen_y],axis=1)
hatalar.columns=["gercek_y","tahminedilen_y"]
hatalar["hata"]=hatalar["gercek_y"]-hatalar["tahminedilen_y"]
hatalar["hata_kareler"]=hatalar["hata"]**2
hatalar["hata_kareler_ort"]=np.mean(hatalar["hata_kareler"])
print(hatalar)

