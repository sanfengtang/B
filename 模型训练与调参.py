#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import sys
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
pd.set_option('display.max_rows', 100,'display.max_columns', 1000,"display.max_colwidth",1000,'display.width',1000)

from sklearn.metrics import *
from sklearn.linear_model import *
from sklearn.neighbors import *
from sklearn.svm import *
from sklearn.neural_network import *
from sklearn.tree import *
from sklearn.ensemble import *
from xgboost import *
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import *
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import *


# # 读取数据

# In[ ]:


data = pd.read_excel("final_data.xlsx", na_values=np.nan)


# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


data.describe()


# # 去除0-1数据以及One-Hot数据后的列

# In[ ]:


corr_cols = list(data.columns[:28]) + list(data.columns[43:49])


# In[ ]:


test_data = data[corr_cols]


# In[ ]:


test_data_corr = test_data.corr()


# In[ ]:


price_corr = dict(test_data_corr.iloc[0])


# In[ ]:


price_corr = sorted(price_corr.items(), key=lambda x: abs(x[1]), reverse=True)


# In[ ]:


price_corr


# In[ ]:


price_corr_cols = [ r[0] for r in price_corr ]


# In[ ]:


price_data = test_data_corr[price_corr_cols].loc[price_corr_cols]


# In[ ]:


price_data.shape


# In[ ]:


plt.figure(figsize=(12, 8))
plt.title("相关系数热力图")
ax = sns.heatmap(price_data, linewidths=0.5, cmap='OrRd', cbar=True)
plt.show()


# In[ ]:


plt.figure(figsize=(15, 10))
plt.title("直方图")
sns.histplot(test_data['售价'], bins = np.arange(0,80,5))
sns.histplot(test_data['新车售价'], bins = np.arange(0,80,5), color="pink")
plt.xlim(-4,100)
x_major_locator=MultipleLocator(5)
plt.gca().xaxis.set_major_locator(x_major_locator)
plt.xlabel("价格", fontdict={"size":12})
plt.ylabel("数量", fontdict={"size":12})
plt.show()


# In[ ]:





# In[ ]:





# # 切分数据

# In[ ]:


X = data[ data.columns[1:] ]
y_reg = data[ data.columns[0] ]


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.3, random_state=42)


# In[ ]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# # 评价指标函数定义

# In[ ]:


def evaluation(model):
    ypred = model.predict(x_test)
    mae = mean_absolute_error(y_test, ypred)
    mse = mean_squared_error(y_test, ypred)
    rmse = sqrt(mse)
    print("MAE: %.2f" % mae)
    print("MSE: %.2f" % mse)
    print("RMSE: %.2f" % rmse)
    return ypred


# # 线性回归模型

# In[ ]:


model_LR = LinearRegression()
model_LR.fit(x_train, y_train)
print("params: ", model_LR.get_params())
print("train score: ", model_LR.score(x_train, y_train))
print("test score: ", model_LR.score(x_test, y_test))
predict_y = evaluation(model_LR)


# In[ ]:


test_y = np.array(y_test)


# In[ ]:


plt.figure(figsize=(10,10))
# 预测值
plt.title('线性回归-真实值预测值对比')
plt.plot(predict_y[:50], 'ro-', label='预测值')
plt.plot(test_y[:50], 'go-', label='真实值')
plt.legend()
plt.show()


# # KNN

# In[ ]:


model_knn = KNeighborsRegressor()
model_knn.fit(x_train, y_train)
print("params: ", model_knn.get_params())
print("train score: ", model_knn.score(x_train, y_train))
print("test score: ", model_knn.score(x_test, y_test))
predict_y = evaluation(model_knn)


# In[224]:


plt.figure(figsize=(10,10))
# 预测值
plt.title('KNN-真实值预测值对比')
plt.plot(predict_y[:50], 'ro-', label='预测值')
plt.plot(test_y[:50], 'go-', label='真实值')
plt.legend()
plt.show()


# # SVM（训练时间太长了）

# In[ ]:


model_svr = SVR()
model_svr.fit(x_train, y_train)
print("params: ", model_svr.get_params())
print("score: ", model_svr.score(x_test, y_test))
evaluation(model_svr)


# # 岭回归

# In[225]:


model_ridge = Ridge()
model_ridge.fit(x_train, y_train)
print("params: ", model_ridge.get_params())
print("train score: ", model_ridge.score(x_train, y_train))
print("test score: ", model_ridge.score(x_test, y_test))
predict_y = evaluation(model_ridge)


# In[226]:


plt.figure(figsize=(10,10))
# 预测值
plt.title('岭回归-真实值预测值对比')
plt.plot(predict_y[:50], 'ro-', label='预测值')
plt.plot(test_y[:50], 'go-', label='真实值')
plt.legend()
plt.show()


# # LASSO回归 （训练时间长）

# In[ ]:


model_lasso = Lasso()
model_lasso.fit(x_train, y_train)
print("params: ", model_lasso.get_params())
print("score: ", model_lasso.score(x_test, y_test))
evaluation(model_lasso)


# # 多层感知机

# In[228]:


model_mlp = MLPRegressor(random_state=42)
model_mlp.fit(x_train, y_train)
print("params: ", model_mlp.get_params())
print("train score: ", model_mlp.score(x_train, y_train))
print("test score: ", model_mlp.score(x_test, y_test))
predict_y = evaluation(model_mlp)


# In[229]:


plt.figure(figsize=(10,10))
# 预测值
plt.title('多层感知机-真实值预测值对比')
plt.plot(predict_y[:50], 'ro-', label='预测值')
plt.plot(test_y[:50], 'go-', label='真实值')
plt.legend()
plt.show()


# # 决策树回归

# In[233]:


model_dtr = DecisionTreeRegressor(max_depth = 5, random_state=30)
model_dtr.fit(x_train, y_train)
print("params: ", model_dtr.get_params())
print("train score: ", model_dtr.score(x_train, y_train))
print("test score: ", model_dtr.score(x_test, y_test))
predict_y = evaluation(model_dtr)


# In[234]:


model_dtr.get_depth()


# In[230]:


model_dtr = DecisionTreeRegressor( random_state=30)
model_dtr.fit(x_train, y_train)
print("params: ", model_dtr.get_params())
print("train score: ", model_dtr.score(x_train, y_train))
print("test score: ", model_dtr.score(x_test, y_test))
predict_y = evaluation(model_dtr)


# In[231]:


model_dtr.get_depth()


# In[235]:


plt.figure(figsize=(10,10))
# 预测值
plt.title('决策树回归-真实值预测值对比')
plt.plot(predict_y[:50], 'ro-', label='预测值')
plt.plot(test_y[:50], 'go-', label='真实值')
plt.legend()
plt.show()


# # 极限树回归

# In[195]:


model_etr = ExtraTreeRegressor(random_state=30)
model_etr.fit(x_train, y_train)
print("params: ", model_etr.get_params())
print("train score: ", model_etr.score(x_train, y_train))
print("test score: ", model_etr.score(x_test, y_test))
predict_y = evaluation(model_etr)


# In[196]:


model_etr.get_depth() # 树太深了，过拟合


# In[197]:


plt.figure(figsize=(10,10))
# 预测值
plt.title('极限树回归-真实值预测值对比')
plt.plot(predict_y[:50], 'ro-', label='预测值')
plt.plot(test_y[:50], 'go-', label='真实值')
plt.legend()
plt.show()


# # 随机森林

# In[ ]:


model_rfr = RandomForestRegressor(random_state=30)
model_rfr.fit(x_train, y_train)
print("params: ", model_rfr.get_params())
print("train score: ", model_rfr.score(x_train, y_train))
print("test score: ", model_rfr.score(x_test, y_test))
predict_y = evaluation(model_rfr)


# In[ ]:


feature_important = sorted(
    zip(x_train.columns, map(lambda x:round(x,4), model_rfr.feature_importances_)),
    key=lambda x: x[1],reverse=True)

for i in range(33):
    print(feature_important[i])


# In[ ]:


f1_list = []
f2_list = []

for i in range(33):
    f1_list.append(feature_important[i][0])

for i in range(1, 34):
    f2_list.append(price_corr[i][0])
    
cnt = 0
for i in range(33):
    if f1_list[i] in f2_list:
        print(f1_list[i])
        cnt += 1
print("共有"+str(cnt)+"个重复特征！")


# In[ ]:


plt.figure(figsize=(10,10))
# 预测值
plt.title('随机森林-真实值预测值对比')
plt.plot(predict_y[:50], 'ro-', label='预测值')
plt.plot(test_y[:50], 'go-', label='真实值')
plt.legend()
plt.show()


# # Adaboost

# In[ ]:


model_abr = AdaBoostRegressor()
model_abr.fit(x_train, y_train)
print("params: ", model_abr.get_params())
print("train score: ", model_abr.score(x_train, y_train))
print("test score: ", model_abr.score(x_test, y_test))
predict_y = evaluation(model_abr)


# In[ ]:


plt.figure(figsize=(10,10))
# 预测值
plt.title('Adaboost-真实值预测值对比')
plt.plot(predict_y[:50], 'ro-', label='预测值')
plt.plot(test_y[:50], 'go-', label='真实值')
plt.legend()
plt.show()


# # 梯度提升树

# In[ ]:


model_gbr = GradientBoostingRegressor()
model_gbr.fit(x_train, y_train)
print("params: ", model_gbr.get_params())
print("train score: ", model_gbr.score(x_train, y_train))
print("test score: ", model_gbr.score(x_test, y_test))
predict_y = evaluation(model_gbr)


# In[ ]:


plt.figure(figsize=(10,10))
# 预测值
plt.title('梯度提升树-真实值预测值对比')
plt.plot(predict_y[:50], 'ro-', label='预测值')
plt.plot(test_y[:50], 'go-', label='真实值')
plt.legend()
plt.show()


# # Bagging

# In[ ]:


model_br = BaggingRegressor(random_state=30)
model_br.fit(x_train, y_train)
print("params: ", model_br.get_params())
print("train score: ", model_br.score(x_train, y_train))
print("test score: ", model_br.score(x_test, y_test))
predict_y = evaluation(model_br)


# In[ ]:


plt.figure(figsize=(10,10))
# 预测值
plt.title('Bagging-真实值预测值对比')
plt.plot(predict_y[:50], 'ro-', label='预测值')
plt.plot(test_y[:50], 'go-', label='真实值')
plt.legend()
plt.show()


# # XGBR

# In[ ]:


model_xgbr = XGBRegressor(n_estimators = 200, max_depth=5, random_state=1024)
model_xgbr.fit(x_train, y_train)
print("params: ", model_xgbr.get_params())
print("train score: ", model_xgbr.score(x_train, y_train))
print("test score: ", model_xgbr.score(x_test, y_test))
predict_y = evaluation(model_xgbr)


# In[252]:


plt.figure(figsize=(10,10))
# 预测值
plt.title('XGBR-真实值预测值对比')
plt.plot(predict_y[:50], 'ro-', label='预测值')
plt.plot(test_y[:50], 'go-', label='真实值')
plt.legend()
plt.show()


# # LGBM

# In[ ]:


model_lgb = lgb.LGBMRegressor(num_leaves=40, max_depth=5, random_state=42)
model_lgb.fit(x_train, y_train)
print("params: ", model_lgb.get_params())
print("train score: ", model_lgb.score(x_train, y_train))
print("test score: ", model_lgb.score(x_test, y_test))
predict_y = evaluation(model_lgb)


# In[ ]:


plt.figure(figsize=(10,10))
# 预测值
plt.title('LGBM-真实值预测值对比')
plt.plot(predict_y[:50], 'ro-', label='预测值')
plt.plot(test_y[:50], 'go-', label='真实值')
plt.legend()
plt.show()


# # 集成模型voting

# In[ ]:


model_voting = VotingRegressor(estimators=[('model_LR', model_LR), 
                                           ('model_knn', model_knn), 
                                                ('model_dtr', model_dtr),
                                                ('model_rfr', model_rfr),
                                                ('model_xgbr', model_xgbr)
                                            ])
model_voting.fit(x_train, y_train)
# print("params: ", model_voting.get_params())
print("train score: ", model_voting.score(x_train, y_train))
print("test score: ", model_voting.score(x_test, y_test))
predict_y = evaluation(model_voting)


# In[ ]:


plt.figure(figsize=(10,10))
# 预测值
plt.title('集成模型voting-真实值预测值对比')
plt.plot(predict_y[:50], 'ro-', label='预测值')
plt.plot(test_y[:50], 'go-', label='真实值')
plt.legend()
plt.show()


# # Tensorflow 神经网络

# In[ ]:


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[ ]:


train_x = np.array(x_train_scaled)
#train_x_ = train_x_[:900]
#val_x = train_x_[900:]

test_x = np.array(x_test_scaled)


train_y = np.array(y_train)
# train_y = train_y[:900]
# val_y = train_y[900:]

test_y = np.array(y_test)


# In[ ]:


# 1
model_tf = tf.keras.Sequential()

# 2
# layer1 = layers.Dense(128, activation='relu')
# layer1.get_weights()
#model.add(layers.Dense(47, activation='relu'))
# model_tf.add(layers.Dense(256, activation='relu'))
model_tf.add(layers.Dense(128, activation='relu'))
model_tf.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(16, activation='relu'))
model_tf.add(layers.Dense(8, activation='relu'))
model_tf.add(layers.Dense(4, activation='relu'))
model_tf.add(layers.Dense(1, activation='relu'))
model_tf.build(input_shape =(None,189))
model_tf.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
             loss=tf.keras.losses.mean_squared_error,
             metrics=['mse', 'mae']) #, tf.keras.metrics.mse, tf.keras.metrics.RootMeanSquaredError])
# 3
# model.build(input_shape =(None,47))
# model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
#              loss=tf.keras.losses.categorical_crossentropy,
#              metrics=[tf.keras.metrics.categorical_accuracy])

# 4
history = model_tf.fit(train_x, train_y, epochs=200, batch_size=128,
                    #validation_data=(train_x_[2800:], train_y[2800:]),shuffle=False)
                    validation_split = 0.2, #从测试集中划分80%给训练集
                    validation_freq = 1) #测试的间隔次数为20, validation_data=(val_x, val_y))

# 5
model_tf.summary()


# In[ ]:


history_dict = history.history
loss_values = history_dict['loss']
mae=history_dict["mae"]
mse=history_dict["mse"]
val_loss = history_dict['val_loss']
val_mae = history_dict['val_mae']
val_mse = history_dict['val_mse']
epochs = range(1, len(loss_values) + 1)

plt.figure(figsize=(8,5))
plt.plot(range(1, len(loss_values) + 1), loss_values, label = 'Training loss')
plt.plot(range(1, len(val_loss) + 1), val_loss, label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#plt.plot(loss_values)
#plt.plot(val_loss)
#plt.plot(pre)
plt.show()


# In[ ]:


plt.figure(figsize=(8,5))
plt.plot(range(1, len(mae) + 1), mae, label = 'mae')
plt.plot(range(1, len(val_mae) + 1), val_mae, label = 'val_mae')
plt.title('mean_absolute_error')
plt.xlabel('Epochs')
plt.ylabel('error')
plt.legend()
#plt.plot(loss_values)
#plt.plot(val_loss)
#plt.plot(pre)
plt.show()


# In[ ]:


plt.figure(figsize=(8,5))
plt.plot(range(1, len(mse) + 1), mse, label = 'mse')
plt.plot(range(1, len(val_mse) + 1), val_mse, label = 'val_mse')
plt.title('mean_squre_error')
plt.xlabel('Epochs')
plt.ylabel('error')
plt.legend()
#plt.plot(loss_values)
#plt.plot(val_loss)
#plt.plot(pre)
plt.show()


# In[ ]:


ypred = model_tf.predict(test_x)
mae = mean_absolute_error(test_y, ypred)
mse = mean_squared_error(test_y, ypred)
rmse = sqrt(mse)

train_ypred = model_tf.predict(train_x)
train_r2_score = r2_score(train_y, train_ypred)
test_r2_score = r2_score(test_y, ypred)
print("MAE: %.2f" % mae)
print("MSE: %.2f" % mse)
print("RMSE: %.2f" % rmse)
print("train score: ", train_r2_score)
print("test score: ", test_r2_score)


# In[ ]:





# In[ ]:





# In[ ]:


def evaluation_tf(model):
    ypred = model.predict(test_x)
    mae = mean_absolute_error(test_y, ypred)
    mse = mean_squared_error(test_y, ypred)
    rmse = sqrt(mse)
    print("MAE: %.2f" % mae)
    print("MSE: %.2f" % mse)
    print("RMSE: %.2f" % rmse)


# In[ ]:


evaluation_tf(model_tf)


# In[ ]:


model_tf.evaluate(test_x, test_y, verbose=0)


# In[ ]:


model_tf.metrics_names


# In[ ]:


predict_y = model_tf.predict(test_x)
[*zip(test_y, predict_y)]


# In[ ]:





# In[ ]:





# In[ ]:




