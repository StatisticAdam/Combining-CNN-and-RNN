from keras.datasets import imdb
from keras.preprocessing import  sequence
from keras.utils import pad_sequences
import numpy as np
np.set_printoptions(threshold=np.inf)


max_features=10000
max_len=500
print('loading data...')
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=max_features
                                                 )
print(len(x_train))
print(len(x_test))
print('pad sequences (samples x time)')
x_train=pad_sequences(x_train,maxlen=max_len)
x_test=pad_sequences(x_test,maxlen=max_len)
print('x_train',x_train.shape)
print('x_test',x_test.shape)

#一维卷积的构架与二维相同，它是Conv1D层和MaxPoolong1D层的堆叠，最后是一个全局池化层或Flatten层，将三维输出转换为二维输出，可以向模型中添加一个或多个Dense层

from keras.models import Sequential#用于构建线性堆叠神经网络模型
from keras import layers
from keras.optimizers import  RMSprop #优化器，负责根据损失函数的梯度调整网络参数，以最小化损失
model=Sequential()
model.add(layers.Embedding(max_features,128,input_length=max_len))
model.add(layers.Conv1D(32,7,activation='relu'))
model.add(layers.MaxPool1D(5))
model.add(layers.Conv1D(32,7,activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(1))

model.summary()
model.compile(optimizer=RMSprop(lr=1e-4),loss='binary_crossentropy',metrics=['acc'])
history=model.fit(x_train,y_train,epochs=4,batch_size=128,validation_split=0.2)

import matplotlib.pyplot as plt
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label='T acc')
plt.plot(epochs,val_acc,'b',label='V acc')
plt.title('T and V acc')
plt.legend()
plt.show()

