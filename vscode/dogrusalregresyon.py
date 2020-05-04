import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv(r"C:\Users\DELL\Desktop\python\vscode\Advertising.csv")
df=df.iloc[:,1:len(df)]
print(df.head())
print(df.info())
import seaborn as sns
a=sns.jointplot(x="TV", y="sales", data=df, kind="reg")
b=sns.jointplot(x="radio", y="sales" , data=df, kind="reg")
##plt.show(a), plt.show(b)

from sklearn.linear_model import LinearRegression
X=df[["TV"]]
print(X.head())
y=df[["sales"]]
reg=LinearRegression()
model=reg.fit(X,y)
b0=model.intercept_
ksayı=model.coef_
model=model.score(X,y)  ## rkare, 
print(b0,ksayı,model)
