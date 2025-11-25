import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


X_train = np.load('./binary_X_train.npy', allow_pickle=True)
X_test = np.load('./binary_X_test.npy', allow_pickle=True)
Y_train = np.load('./binary_Y_train.npy', allow_pickle=True)
Y_test = np.load('./binary_Y_test.npy', allow_pickle= True)
print(X_train.shape,Y_train.shape)
print(X_test.shape, Y_test.shape)

model= Sequential()
model.add(Conv2D(32,input_shape=(64,64,3), kernel_size=(3,3), padding= 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2), padding= 'same'))
model.add(Conv2D(64, kernel_size=(3,3), padding= 'same', activation= 'relu'))
model.add(MaxPooling2D(pool_size=(2,2), padding= 'same'))
model.add(Conv2D(128,kernel_size=(3,3), padding= 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2), padding= 'same'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256,activation= 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation= 'sigmoid'))
model.summary()

model.compile(loss= 'binary_crossentropy', optimizer = 'adam', metrics= ['accuracy'])
early_stopping = EarlyStopping(monitor= 'val_accuracy',patience=7)  #얼리스타핑 객체 검증어큐러시로 해서 7에폭까지 검증이 더 좋아지지 않아도 참다가 더 좋아지지 않으면 멈춘다.
                           #벨류데이션이 가장 높았던 7번째 전에 것이 저장이 된다. 떨어지기 전까지만 학습이 되는 와중에 자기가 알아서 멈춰서
fit_hist = model.fit(X_train, Y_train, batch_size=64, epochs= 1000, validation_split=0.15, callbacks=[early_stopping])

score = model.evaluate(X_test,Y_test)
print('Evaluation loss:', score[0])
print('Evaluation accuracy:', score[1])

model.save('./cat_and_dog_binary_classification2_{}.h5'.format(score[1]))
plt.plot(fit_hist.history['loss'], label= 'loss')
plt.plot(fit_hist.history['val_loss'], label= 'validation loss')
plt.legend()
plt.show()

plt.plot(fit_hist.history['accuracy'], label= 'train accuracy')
plt.plot(fit_hist.history['val_accuracy'], label= 'validation accuracy')
plt.legend()
plt.show()
