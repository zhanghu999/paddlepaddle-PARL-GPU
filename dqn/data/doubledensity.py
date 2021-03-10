import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TwoNomal():
    def __init__(self,mu1,mu2,sigma1,sigma2):
        self.mu1 = mu1
        self.sigma1 = sigma1
        self.mu2 = mu2
        self.sigma2 = sigma2
    def doubledensity(self,x):
            mu1 = self.mu1
            sigma1 = self.sigma1
            mu2 = self.mu2
            sigma2 = self.sigma1
            N1 = np.sqrt(2 * np.pi * np.power(sigma1, 2))
            fac1 = np.power(x - mu1, 2) / np.power(sigma1, 2)
            density1=np.exp(-fac1/2)/N1

            N2 = np.sqrt(2 * np.pi * np.power(sigma2, 2))
            fac2 = np.power(x - mu2, 2) / np.power(sigma2, 2)
            density2=np.exp(-fac2/2)/N2
            #print(density1,density2)
            density=(0.5*density2+0.5*density1)*1000
            return density





N2 = TwoNomal(10,80,10,10)

#创建等差数列作为X
X = np.arange(0,100,1)
#print(X)
Y = N2.doubledensity(X)

plt.plot(X,Y,'b-',linewidth=3)

plt.show()

'''数组写入excel表'''
def excelWriter(A):
    data = pd.DataFrame(A)
    writer = pd.ExcelWriter('A.xlsx')  # 写入Excel文件
    data.to_excel(writer, 'page_1', float_format='%.5f')  # ‘page_1’是写入excel的sheet名
    writer.save()
    writer.close()


print(Y)
excelWriter(Y)


