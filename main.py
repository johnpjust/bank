import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from sklearn import preprocessing, metrics
import xgboost

############## one-hot encoding function ###############
def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res)

############ read and preprocess data ##################
df = pd.read_csv(r'C:\Users\justjo\Desktop\bank_nopar.csv', sep=',')

cols_to_encode = ['job', 'marital', 'education', 'default', 'housing',
       'loan', 'contact','month','poutcome']

for col in cols_to_encode:
    df = encode_and_bind(df, col)

val_col = df.Validation.copy()
labels = df.y.copy()
reweight = df.reweight.copy()
df = df.drop('Validation', axis=1)
df = df.drop('y', axis=1)
df = df.drop('reweight', axis=1)

lb = preprocessing.LabelBinarizer()

train = df[val_col == 'Training']
train_labels = lb.fit_transform(labels[val_col == 'Training'])
train_weights = reweight[val_col == 'Training']
val = df[val_col == 'Validation']
val_labels = lb.fit_transform(labels[val_col == 'Validation'])
val_weights = reweight[val_col == 'Validation']
test = df[val_col == 'Test']
test_labels = lb.fit_transform(labels[val_col == 'Test'])
test_weights = reweight[val_col == 'Test']

labels = lb.fit_transform(labels)
labels = labels.astype(np.float32)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')
# ############### outlier investigation via isolation forest ####################
# ## note:  no outliers removed for modeling
# iForest = IsolationForest(max_samples = 1000, random_state = 42)
# iForest.fit(df)
# iForest_pred = iForest.predict(df)
# iForest_pred = pd.DataFrame(iForest_pred, columns = ['Top'])
#
# print("Number of Outliers:", iForest_pred[iForest_pred['Top'] == -1].shape[0]) ## 4260
# print("Number of rows without outliers:", iForest_pred[iForest_pred['Top'] == 1].shape[0])
#
# ##################################### neural network ############################################
# def df_to_dataset(dataframe, labels, shuffle=True, batch_size=32):
#     dataframe = dataframe.copy()
#     # labels = dataframe.pop('target')
#     ds = tf.data.Dataset.from_tensor_slices((dataframe.values.astype(np.float32), labels.astype(np.float32)))
#     if shuffle:
#         ds = ds.shuffle(buffer_size=len(dataframe))
#     ds = ds.batch(batch_size)
#     return ds
#
# batch_size = 50000
# train_ds = df_to_dataset(train, train_labels, batch_size=batch_size)
# val_ds = df_to_dataset(val, val_labels, shuffle=False, batch_size=batch_size)
# test_ds = df_to_dataset(test, test_labels, shuffle=False, batch_size=batch_size)
#
# actfun = tf.nn.tanh
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(4, activation=actfun, input_dim=train.shape[1]))
# # model.add(tf.keras.layers.Dense(128, activation=actfun))
# # model.add(tf.keras.layers.Dense(32, activation=actfun))
# # model.add(tf.keras.layers.Dense(4, activation=actfun))
# # model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
# model.add(tf.keras.layers.Dense(1, activation=None)) ## logits
#
# earl = tf.keras.callbacks.EarlyStopping(monitor='customLoss', min_delta=0, patience=20, verbose=1, mode='auto', baseline=None, restore_best_weights=True) #binary_accuracy  val_loss
# # tens = tf.keras.callbacks.TensorBoard(log_dir=r'C:\Users\justjo\Downloads\bank\ANN/logs', histogram_freq=0, write_graph=False, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
# # # tensorboard --logdir=C:\Users\justjo\Downloads\bank\ANN/logs
#
# def customLoss(yTrue,yPred):
#     return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(yTrue, yPred, pos_weight=10))  ## upweight pos error due to 9:1 ratio of neg:pos samples
#
# def customMonitor(yTrue,yPred):
#     out = tf.nn.sigmoid(yPred)
#     temp = tf.keras.metrics.binary_accuracy(yTrue, out)
#     return tf.reduce_mean(temp)
#
# # model.compile(optimizer='adam',
# #               loss=customLoss, #'binary_crossentropy'
# #               metrics=[tf.keras.metrics.binary_accuracy])
#
# model.compile(optimizer='adam',
#               loss=customLoss, #'binary_crossentropy'
#               metrics=[customLoss, customMonitor]) #accuracy
#
# model.fit(train_ds,
#           validation_data=val_ds,
#           epochs=5000,
#           callbacks=[earl])
#
# out = tf.nn.sigmoid(model(df.values.astype(np.float32)))
# binacc=tf.keras.metrics.binary_accuracy(labels, out)
# print(np.mean(binacc))
#
# binacc_pos=tf.keras.metrics.binary_accuracy(labels[labels==1], out[labels==1])
# print(np.mean(binacc_pos))
#
# # loss, accuracy = model.evaluate(test_ds)
# # print("Accuracy", accuracy)

########################### xgboost ############################################
##### boosted tree
# xgb = xgboost.XGBClassifier(max_depth=8, scale_pos_weight=9, n_estimators=1000) ##n_estimators=1000, learning_rate=0.05
# xgb.fit(X=train.values.astype(np.float32), y=np.squeeze(train_labels.astype(np.float32)), early_stopping_rounds=20,
#         eval_set=[(val.values.astype(np.float32), val_labels.astype(np.float32))], verbose=True)
#
# # make predictions
# predxgb = xgb.predict(test.values.astype(np.float32))
# xgb_conf_mat = metrics.confusion_matrix(test_labels.astype(np.float32), predxgb) ## tree methods tend to have higher false negative rates than ANN
# print(xgb_conf_mat/np.expand_dims(np.sum(xgb_conf_mat, axis=1), axis=1))

# xgb.feature_importances_

#### random forest
xgbrf = xgboost.XGBRFClassifier(max_depth=8, scale_pos_weight=9, n_estimators=100)
# xgbrf = xgboost.XGBRFClassifier(scale_pos_weight=9)
xgbrf.fit(X=train.values.astype(np.float32), y=np.squeeze(train_labels.astype(np.float32)), early_stopping_rounds=20,
        eval_set=[(val.values.astype(np.float32), val_labels.astype(np.float32))], verbose=True)
predxgbrf = xgbrf.predict(test.values.astype(np.float32))
xgbrf_conf_mat = metrics.confusion_matrix(test_labels.astype(np.float32), predxgbrf) ## tree methods tend to have higher false negative rates than ANN
print(xgbrf_conf_mat/np.expand_dims(np.sum(xgbrf_conf_mat, axis=1), axis=1))

# xgbrf.feature_importances_