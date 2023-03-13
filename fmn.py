from tensorflow import keras
fashionn_mnist=keras.datasets.fashion_mnist
(xtr,ytr),(xts,yts)=fashionn_mnist.load_data()
xvalid,xtrain=xtr[:5000]/255.0,xtr[5000:]/255.0
yvalid,ytrain=ytr[:5000]/255.0,ytr[5000:]/255.0
model=keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300,activation="relu"))
model.add(keras.layers.Dense(100,activation="relu"))
model.add(keras.layers.Dense(10,activation="softmax"))
model.summary()
model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])
hist=model.fit(xtrain,ytrain,epochs=50,validation_data=(xvalid,yvalid))