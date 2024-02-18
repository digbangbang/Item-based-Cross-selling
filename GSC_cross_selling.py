# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 09:31:25 2021

@author: abzfxzz
"""

import pandas as pd
import numpy as np
import math
from collections import defaultdict
from operator import itemgetter


data=pd.read_excel('C:/Users/ABZFXZZ/Desktop/EMD - Cross Selling.xlsx',sheet_name='Sheet1')
data=data[data['Qty']>0]


#SKU市场份额
# SKU_amount=data.groupby('SKU').Amount.sum().sort_values(ascending=False)
# print('销售额前50的SKU所占份额:')
# print(SKU_amount[:50].sum()/SKU_amount.sum())
# print('销售额前100的SKU所占份额:')
# print(SKU_amount[:100].sum()/SKU_amount.sum())

# y=SKU_amount[:100]
# plt.rcParams['figure.figsize']=(40,40)
# color=plt.cm.cool(np.linspace(0,1,100))
# squarify.plot(sizes=y.values,label=y.index,alpha=0.8,color=color)
# plt.title('Tree Map for Popular SKU')
# plt.axis('off')
# plt.savefig('Tree_Map_for_Popular_SKU.png')

################################################################
data_all=data[['GSC','SKU','InAcc ID','InAcc Province','InAcc Type','Amount']]
##basket analysis two whole
data_all=data_all.sort_values('GSC')
SKU_columns=sorted(data_all['GSC'].value_counts().index)
inacc_1=[]
inacc_2=[]
ALL_acc=len(data_all['InAcc ID'].unique())
two_basket={}
for i in SKU_columns[:65]:
    data_1=data_all[data_all['GSC']==i]
    inacc_1=data_1['InAcc ID'].unique()
    for j in SKU_columns[SKU_columns.index(i)+1:]:
        data_2=data_all[data_all['GSC']==j]
        inacc_2=data_2['InAcc ID'].unique()
        i_and_j=len(list(set(inacc_1) & set(inacc_2)))
        two_basket.setdefault(i+' and '+j,0)
        two_basket[i+' and '+j]=i_and_j/ALL_acc
two_basket_final=sorted(two_basket.items(),key=lambda item : item[1],reverse=True)
two=pd.DataFrame(two_basket_final)
two.to_csv('twobasket_GSC.csv',encoding='utf_8_sig')

# ##basket analysis three 100
SKU_amount=data.groupby('GSC').Amount.sum().sort_values(ascending=False)
y=list(SKU_amount[:100].index)
inacc_1=[]
inacc_2=[]
inacc_3=[]
three_basket={}
for i in y:
    data_1=data_all[data_all['GSC']==i]
    inacc_1=data_1['InAcc ID'].unique()
    for j in y[y.index(i)+1:]:
        data_2=data_all[data_all['GSC']==j]
        inacc_2=data_2['InAcc ID'].unique()
        for k in y[y.index(j)+1:]:
            data_3=data_all[data_all['GSC']==k]
            inacc_3=data_3['InAcc ID'].unique()
            i_and_j_and_k=len(list(set(inacc_1) & set(inacc_2) & set(inacc_3)))
            three_basket.setdefault(i+' and '+j+' and '+k,0)
            three_basket[i+' and '+j+' and '+k]=i_and_j_and_k/ALL_acc
three_basket_final=sorted(three_basket.items(),key=lambda item : item[1],reverse=True)
three=pd.DataFrame(three_basket_final)
three.to_csv('threebasket_GSC.csv',encoding='utf_8_sig')

################################################################
# ##cross-selling

train=[]
for idx,row in data.iterrows():
    ID=str(row['InAcc ID'])
    SKU=str(row['GSC'])
    train.append([ID,SKU])
traindata={}
for ID,SKU in train:
    traindata.setdefault(ID,set())
    traindata[ID].add(SKU)############倒排表

SKU_columns=sorted(data['GSC'].unique())
InAcc_ID_columns=sorted([int(key) for key in traindata.keys()])
InAcc_ID_columns=[str(x) for x in InAcc_ID_columns]
trans=np.zeros([len(InAcc_ID_columns),len(SKU_columns)])
for i,j in traindata.items():
    m=list(j)
    m.sort()
    amount=data[data['InAcc ID']==int(i)].groupby('GSC').Amount.sum().sort_index()
    for x in j:
        trans[InAcc_ID_columns.index(i)][SKU_columns.index(x)]=amount[m.index(x)]######interest level
            
tran=pd.DataFrame(trans,columns=SKU_columns,index=InAcc_ID_columns)
N=defaultdict(int)
SKUSimMatrix=dict()
for ID,SKU in traindata.items():
    for i in SKU:
        SKUSimMatrix.setdefault(i,dict())
        N[i]+=1
        for j in SKU:
            if i==j:
                continue
            SKUSimMatrix[i].setdefault(j,0)
            SKUSimMatrix[i][j]+=1######共现矩阵
for i,related_SKU in SKUSimMatrix.items():
    for j,cij in related_SKU.items():
        SKUSimMatrix[i][j]=cij/math.sqrt(N[i]*N[j])#######相似度矩阵
##################直接用下面recommen
# recommends=dict()
# for i in range(len(tran)):
#     i_ID_SKU=tran.iloc[i]
#     recommends.setdefault(tran.iloc[i].name,dict())
#     for SKU,related_SKU in SKUSimMatrix.items():
#         recommends[tran.index[i]].setdefault(SKU,0)
#         for a,b in related_SKU.items():
#             recommends[tran.index[i]][SKU]+=i_ID_SKU[a]*b

##############imp
# N_imp=defaultdict(int)
# SKUSimMatrix_imp=dict()
# for ID,SKU in traindata.items():
#     for i in SKU:
#         SKUSimMatrix_imp.setdefault(i,dict())
#         N_imp[i]+=1
#         for j in SKU:
#             if i==j:
#                 continue
#             SKUSimMatrix_imp[i].setdefault(j,0)
#             SKUSimMatrix_imp[i][j]+=1/math.log(1+len(SKU))
# for i,related_SKU in SKUSimMatrix_imp.items():
#     for j,cij in related_SKU.items():
#         SKUSimMatrix_imp[i][j]=cij/math.sqrt(N_imp[i]*N_imp[j])
        
# recommends_imp=dict()
# for i in range(len(tran)):
#     i_ID_SKU=tran.iloc[i]
#     recommends_imp.setdefault(tran.iloc[i].name,dict())
#     for SKU,related_SKU in SKUSimMatrix_imp.items():
#         recommends_imp[tran.index[i]].setdefault(SKU,0)
#         for a,b in related_SKU.items():
#             recommends_imp[tran.index[i]][SKU]+=i_ID_SKU[a]*b

#################recommend生成推荐列表
def recommend(user,K,N,skusimmatrix,tran):
    rank={}
    bought_sku=tran.loc[user]
    bought_sku_nu=dict()
    for sku,value in bought_sku.items():
        if value!=0:
            bought_sku_nu.setdefault(sku)
            bought_sku_nu[sku]=value
    for sku,value in bought_sku_nu.items():
        for SKU,related_SKU in sorted(skusimmatrix[sku].items(),key=itemgetter(1),reverse=True)[:K]:
            if SKU in bought_sku_nu:
                continue
            rank.setdefault(SKU,0)
            rank[SKU]+=related_SKU*float(value)
    midd=sorted(rank.items(),key=itemgetter(1),reverse=True)[:N]
    for i in range(3):
        if len(midd)!=3:
            midd.append(())
    return midd
recommends={}
acc_ID=tran.index
for ID in acc_ID:
    recommends.setdefault(ID,())
    recommends[ID]=recommend(ID,705,3,SKUSimMatrix,tran)
recommends_final={}        
for ID,SKU in recommends.items():
    recommends_final.setdefault(ID,())
    m=''
    for i in SKU:
        i=str(i)
        m+=i
    recommends_final[ID]=m
############整理数据
df=pd.DataFrame.from_dict(recommends_final,orient='index',columns=['GSC Recommendation Top3'])
df=df.reset_index().rename(columns={'index':'InAcc ID'})
df1=data_all.groupby('InAcc ID').Amount.sum()
amount=[]
for i in df1:
    amount.append(i)
df.insert(1,'2020 total amount',amount)
df2=data_all.groupby(['InAcc ID','GSC']).agg({'Amount':sum})
df2=df2['Amount'].groupby(level=0,group_keys=False)
df2=df2.nlargest(3)
df2.columns=['InAcc ID','GSC','Amount']
df2.to_csv('df2.csv',encoding='utf_8_sig')
df2=pd.read_csv('df2.csv')
product_3=[]
product_3_pct=[]
pro=df2.iloc[0]['GSC']
pro_pct=df2.iloc[0]['Amount']
for i in range(len(df2)-1):
    if df2.iloc[i]['InAcc ID']!=df2.iloc[i+1]['InAcc ID']:
        pro=str(pro)
        product_3.append(pro)
        product_3_pct.append(pro_pct)
        pro=df2.iloc[i+1]['GSC']
        pro_pct=df2.iloc[i+1]['Amount']
    else:
        pro=pro+','+df2.iloc[i+1]['GSC']
        pro_pct=pro_pct+df2.iloc[i+1]['Amount']
pro=df2.iloc[len(df2['Amount'])-3]['GSC']+','+df2.iloc[len(df2['Amount'])-2]['GSC']+','+df2.iloc[len(df2['Amount'])-1]['GSC']
product_3.append(pro)
pro_pct=df2.iloc[len(df2['Amount'])-3]['Amount']+df2.iloc[len(df2['Amount'])-2]['Amount']+df2.iloc[len(df2['Amount'])-1]['Amount']
product_3_pct.append(pro_pct)
df.insert(2,'GSC Top3',product_3)
df.insert(3,'GSC Top3 pct',product_3_pct)
df['GSC Top3 pct']=df['GSC Top3 pct']/df['2020 total amount']
df.to_csv('df.csv',encoding='utf_8_sig')

# datause=data[['Comm','GSC','SKU Desc']]
# datause_Comm=datause.drop_duplicates()
# datause_Comm=datause_Comm.sort_values(by=['Comm','GSC'],ascending=(True,True))
# datause_Comm['cluster']=datause_Comm['Comm']

# sign=0
# for i in range(len(datause_Comm)-1):
#     datause_Comm.iloc[i,3]=str(datause_Comm.iloc[i,0])+'--'+str(sign)
#     if datause_Comm.iloc[i,1] != datause_Comm.iloc[i+1,1]:
#         sign=sign+1
#     if datause_Comm.iloc[i,0] != datause_Comm.iloc[i+1,0]:
#         sign=0
# datause_Comm.iloc[len(datause_Comm)-1,3]='7210--0'
# data_GSC=datause_Comm[['SKU Desc','cluster']]
# data_GSC.to_csv('GSC_SKU.csv',encoding='utf_8_sig')













