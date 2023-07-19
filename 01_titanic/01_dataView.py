# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:08:15 2023

@author: sun78
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号


# 获取当前文件夹下的所有文件名
files = [f for f in os.listdir('.') if os.path.isfile(f)]

# 打印文件名（不包括子文件夹）
for file in files:
    print(file)
print('\n')


# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
# 设置1000列的时候才换行
pd.set_option('display.width', 1000)


train = pd.read_csv('train.csv')
# print(train)

print(train.info(), '\n')

print(train.describe(), '\n')



fig = plt.figure()
fig.set(alpha = 0.5)

# 在一张大图里分列几个小图
plt.subplot2grid((2,3),(0,0))

train.Survived.value_counts().plot(kind = 'bar') # 柱状图
plt.title(u"获救情况1为获救")
plt.ylabel(u'人数')

plt.subplot2grid((2,3),(0,1))
train.Pclass.value_counts().plot(kind="bar")
plt.ylabel(u"人数")
plt.title(u"乘客等级分布")

plt.subplot2grid((2,3),(0,2))
plt.scatter(train.Survived, train.Age)
plt.ylabel(u"年龄")                         # 设定纵坐标名称
plt.grid(b=True, which='major', axis='y')
plt.title(u"按年龄看获救分布 (1为获救)")


plt.subplot2grid((2,3),(1,0), colspan=2)
train.Age[train.Pclass == 1].plot(kind='kde')
train.Age[train.Pclass == 2].plot(kind='kde')
train.Age[train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"年龄")# plots an axis lable
plt.ylabel(u"密度")
plt.title(u"各等级的乘客年龄分布")
plt.legend((u'头等舱', u'2等舱',u'3等舱'),loc='best') # sets our legend for our graph.


plt.subplot2grid((2,3),(1,2))
train.Embarked.value_counts().plot(kind='bar')
plt.title(u"各登船口岸上船人数")
plt.ylabel(u"人数")
Survived_0 = train.Pclass[train.Survived == 0].value_counts()
Survived_1 = train.Pclass[train.Survived == 1].value_counts()
df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"各乘客等级的获救情况")
plt.xlabel(u"乘客等级")
plt.ylabel(u"人数")

plt.show()


print(df)
