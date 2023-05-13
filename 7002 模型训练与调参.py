#!/usr/bin/env python
# coding: utf-8

# In[2]:


import lightgbm as lgb
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
from math import sqrt


# In[3]:


data = pd.read_excel(r'C:\Users\Administrator\Desktop\2023\7002\data v1\final_data.xlsx')


# In[4]:


data.shape


# In[5]:


data.head()


# In[6]:


data.describe()


# In[7]:


import pandas as pd

# 计算变量间的相关系数矩阵
corr_matrix = data[['front side airbag', 'keyless start system', 'TRC traction control system', 'uphill assistance',
                'Electric sunroof', 'leather steering wheel', 'daytime running lights', 'automatic headlights',
                'mirror heating', 'rear wiper', 'Rear seat air outlet', 'transfer record', '4S shop maintenance',
                'Original car purchase/transfer invoice', 'Vehicle purchase tax payment certificate', 'passenger/person', 'selling price']].corr()

# 选取与价格的相关系数作为指标
price_corr = corr_matrix['selling price'].iloc[:-1]

# 打印结果
print(price_corr)


# In[8]:


# Define the columns to plot and calculate the mean selling price for each
cols = ["front side airbag", "keyless start system", "TRC traction control system", 
        "uphill assistance", "Electric sunroof", "leather steering wheel", 
        "daytime running lights", "automatic headlights", "mirror heating", "rear wiper",
        "Rear seat air outlet", "transfer record", "4S shop maintenance"]

means = []
for col in cols:
    means.append(data[data[col] == 1]["selling price"].mean())

# Plot the bar chart
plt.bar(x=cols, height=means)

# Set the chart title and axis labels
plt.title("Average Selling Price by Feature")
plt.xlabel("Features")
plt.ylabel("Average Selling Price")

# Rotate the x-axis labels for readability
plt.xticks(rotation=90)

# Display the chart
plt.show()


# In[9]:


cat_vars = data.select_dtypes(include=['object']).columns.tolist()


# In[10]:


# Extract numeric variables
float_vars = data.select_dtypes(include=['float']).columns.tolist()

# Calculate the correlation coefficient matrix
corr_matrix = data[float_vars].corr()

# Draw a heat map of the correlation coefficient matrix
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5)

# display graphics
plt. show()


# In[11]:


import seaborn as sns
sns.pairplot(data, vars=float_vars)


# # Column after removing 0-1 data and One-Hot data

# In[12]:


corr_cols = list(data.columns[:28]) + list(data.columns[43:49])


# In[13]:


test_data = data[corr_cols]


# In[14]:


test_data_corr = test_data.corr()


# In[15]:


price_corr = dict(test_data_corr.iloc[0])


# In[16]:


price_corr = sorted(price_corr.items(), key=lambda x: abs(x[1]), reverse=True)


# In[17]:


price_corr


# In[18]:


price_corr_cols = [ r[0] for r in price_corr ]


# In[19]:


price_data = test_data_corr[price_corr_cols].loc[price_corr_cols]


# In[20]:


price_data.shape


# In[21]:


# 对数据集进行描述性统计
price_data.describe()


# In[22]:


plt.figure(figsize=(12, 8))
plt.title("Correlation coefficient heat map")
ax = sns.heatmap(price_data, linewidths=0.5, cmap='OrRd', cbar=True)
plt.show()


# In[23]:


plt.figure(figsize=(15, 10))
plt.title("Histogram")
sns.histplot(test_data['selling price'], bins = np.arange(0,80,5))
sns.histplot(test_data['new car price'], bins = np.arange(0,80,5), color="pink")
plt. xlim(-4,100)
x_major_locator=MultipleLocator(5)
plt.gca().xaxis.set_major_locator(x_major_locator)
plt.xlabel("price", fontdict={"size":12})
plt.ylabel("quantity", fontdict={"size":12})
plt. show()


# # Split data

# In[24]:


X = data[ data.columns[1:] ]
y_reg = data[ data.columns[0] ]


# In[25]:


x_train, x_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.3, random_state=42)


# In[26]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[27]:


# Check the number of effective samples
n_samples = len(data.index)
print("Number of valid samples:", n_samples)

# View the number of independent and dependent variables for each sample
n_features = len(data.columns) - 1 # because the last column is the dependent variable
n_targets = 1 # Assume there is only one dependent variable
print("Number of independent variables per sample:", n_features)
print("Number of dependent variables per sample:", n_targets)


# # Evaluation indicator function definition

# In[28]:


def evaluation(model):
    ypred = model.predict(x_test)
    mae = mean_absolute_error(y_test, ypred)
    mse = mean_squared_error(y_test, ypred)
    rmse = sqrt(mse)
    print("MAE: %.2f" % mae)
    print("MSE: %.2f" % mse)
    print("RMSE: %.2f" % rmse)
    return ypred


# # linear regression model 线性回归模型

# In[29]:


model_LR = LinearRegression()
model_LR.fit(x_train, y_train)
print("params: ", model_LR.get_params())
print("train score: ", model_LR.score(x_train, y_train))
print("test score: ", model_LR.score(x_test, y_test))
predict_y = evaluation(model_LR)


# In[30]:


test_y = np.array(y_test)


# In[31]:


plt.figure(figsize=(10,10))
# Predictive value
plt.title('Linear Regression-Comparison of Predicted Values of Real Values')
plt.plot(predict_y[:100], 'ro-', label='predicted value')
plt.plot(test_y[:100], 'go-', label='real value')
plt. legend()
plt. show()


# In[32]:


import pandas as pd

# Create a table of the first 50 predicted values and real values
df = pd.DataFrame({'Predicted values': predict_y[:50], 'Real values': test_y[:50]})
print(df)


# # KNN

# In[33]:


model_knn = KNeighborsRegressor()
model_knn.fit(x_train, y_train)
print("params: ", model_knn.get_params())
print("train score: ", model_knn.score(x_train, y_train))
print("test score: ", model_knn.score(x_test, y_test))
predict_y = evaluation(model_knn)


# In[34]:


plt.figure(figsize=(10,10))
# Predictive value
plt.title('KNN-true value prediction value comparison')
plt.plot(predict_y[:50], 'ro-', label='predicted value')
plt.plot(test_y[:50], 'go-', label='real value')
plt. legend()
plt. show()


# In[35]:


import numpy as np
import pandas as pd

# Create a 2D array of predicted and real values for the first 50 samples
data = np.column_stack((predict_y[:50], test_y[:50]))

# Convert the 2D array into a Pandas DataFrame
df = pd.DataFrame(data, columns=['Predicted values', 'Real values'])

# Display the DataFrame
print(df)


# # Decision tree regression  决策树回归

# In[36]:


model_dtr = DecisionTreeRegressor(max_depth = 5, random_state=30)
model_dtr.fit(x_train, y_train)
print("params: ", model_dtr.get_params())
print("train score: ", model_dtr.score(x_train, y_train))
print("test score: ", model_dtr.score(x_test, y_test))
predict_y = evaluation(model_dtr)


# In[37]:


model_dtr.get_depth()


# In[38]:


model_dtr = DecisionTreeRegressor( random_state=30)
model_dtr.fit(x_train, y_train)
print("params: ", model_dtr.get_params())
print("train score: ", model_dtr.score(x_train, y_train))
print("test score: ", model_dtr.score(x_test, y_test))
predict_y = evaluation(model_dtr)


# In[39]:


model_dtr.get_depth()


# In[40]:


plt.figure(figsize=(10,10))
# Predictive value
plt.title('Decision Tree Regression-Comparison of Real Value and Predicted Value')
plt.plot(predict_y[:50], 'ro-', label='predicted value')
plt.plot(test_y[:50], 'go-', label='real value')
plt. legend()
plt. show()


# # 随机森林 

# In[41]:


model_rfr = RandomForestRegressor(random_state=30)
model_rfr.fit(x_train, y_train)
print("params: ", model_rfr.get_params())
print("train score: ", model_rfr.score(x_train, y_train))
print("test score: ", model_rfr.score(x_test, y_test))
predict_y = evaluation(model_rfr)


# In[42]:


feature_important = sorted(
    zip(x_train.columns, map(lambda x:round(x,4), model_rfr.feature_importances_)),
    key=lambda x: x[1],reverse=True)

for i in range(33):
    print(feature_important[i])


# In[43]:


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
print("Total"+str(cnt)+"repeated features!")


# In[44]:


plt.figure(figsize=(10,10))
# Predictive value
plt.title('Random forest - real value prediction value comparison')
plt.plot(predict_y[:50], 'ro-', label='predicted value')
plt.plot(test_y[:50], 'go-', label='real value')
plt. legend()
plt. show()


# # Adaboost

# In[45]:


model_abr = AdaBoostRegressor()
model_abr.fit(x_train, y_train)
print("params: ", model_abr.get_params())
print("train score: ", model_abr.score(x_train, y_train))
print("test score: ", model_abr.score(x_test, y_test))
predict_y = evaluation(model_abr)


# In[46]:


plt.figure(figsize=(10,10))
# Predictive value
plt.title('Adaboost-true value prediction value comparison')
plt.plot(predict_y[:50], 'ro-', label='predicted value')
plt.plot(test_y[:50], 'go-', label='real value')
plt. legend()
plt. show()


# # Bagging

# In[47]:


model_br = BaggingRegressor(random_state=30)
model_br.fit(x_train, y_train)
print("params: ", model_br.get_params())
print("train score: ", model_br.score(x_train, y_train))
print("test score: ", model_br.score(x_test, y_test))
predict_y = evaluation(model_br)


# In[48]:


plt.figure(figsize=(10,10))
# Predictive value
plt.title('Bagging-true value prediction value comparison')
plt.plot(predict_y[:50], 'ro-', label='predicted value')
plt.plot(test_y[:50], 'go-', label='real value')
plt. legend()
plt. show()


# # LGBM

# In[49]:


model_lgb = lgb.LGBMRegressor(num_leaves=40, max_depth=5, random_state=42)
model_lgb.fit(x_train, y_train)
print("params: ", model_lgb.get_params())
print("train score: ", model_lgb.score(x_train, y_train))
print("test score: ", model_lgb.score(x_test, y_test))
predict_y = evaluation(model_lgb)


# In[50]:


plt.figure(figsize=(10,10))
# Predictive value
plt.title('LGBM-true value prediction value comparison')
plt.plot(predict_y[:50], 'ro-', label='predicted value')
plt.plot(test_y[:50], 'go-', label='real value')
plt. legend()
plt. show()


# # XGBR

# In[51]:


model_xgbr = XGBRegressor(n_estimators = 200, max_depth=5, random_state=1024)
model_xgbr.fit(x_train, y_train)
print("params: ", model_xgbr.get_params())
print("train score: ", model_xgbr.score(x_train, y_train))
print("test score: ", model_xgbr.score(x_test, y_test))
predict_y = evaluation(model_xgbr)


# In[52]:


plt.figure(figsize=(10,10))
# Predictive value
plt.title('XGBR-true value prediction value comparison')
plt.plot(predict_y[:50], 'ro-', label='predicted value')
plt.plot(test_y[:50], 'go-', label='real value')
plt. legend()
plt. show()


# # 集成模型voting

# In[53]:


model_voting = VotingRegressor(estimators=[('model_LR', model_LR), 
                                           ('model_knn', model_knn), 
                                                ('model_dtr', model_dtr),
                                            ])
model_voting.fit(x_train, y_train)
# print("params: ", model_voting.get_params())
print("train score: ", model_voting.score(x_train, y_train))
print("test score: ", model_voting.score(x_test, y_test))
predict_y = evaluation(model_voting)


# In[54]:


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


# In[55]:


plt.figure(figsize=(10,10))
# Predictive value
plt.title('Comparison of integrated model voting-real value predicted value')
plt.plot(predict_y[:50], 'ro-', label='predicted value')
plt.plot(test_y[:50], 'go-', label='real value')
plt. legend()
plt. show()


# In[ ]:





# # Tensorflow 神经网络

# In[56]:


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[57]:


train_x = np.array(x_train_scaled)
#train_x_ = train_x_[:900]
#val_x = train_x_[900:]

test_x = np.array(x_test_scaled)


train_y = np.array(y_train)
# train_y = train_y[:900]
# val_y = train_y[900:]

test_y = np.array(y_test)


# In[58]:


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


# In[122]:


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


# In[123]:


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


# In[124]:


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


# In[125]:


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





# In[126]:


def evaluation_tf(model):
    ypred = model.predict(test_x)
    mae = mean_absolute_error(test_y, ypred)
    mse = mean_squared_error(test_y, ypred)
    rmse = sqrt(mse)
    print("MAE: %.2f" % mae)
    print("MSE: %.2f" % mse)
    print("RMSE: %.2f" % rmse)


# In[127]:


evaluation_tf(model_tf)


# In[128]:


model_tf.evaluate(test_x, test_y, verbose=0)


# In[129]:


model_tf.metrics_names


# In[130]:


predict_y = model_tf.predict(test_x)
[*zip(test_y, predict_y)]


# In[ ]:





# In[ ]:




