import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

(train_images,train_labels),(test_images,test_labels) = tf.keras.datasets.mnist.load_data()

x_train = train_images.reshape(train_images.shape[0],28,28,1)
x_test = test_images.reshape(test_images.shape[0],28,28,1)

input_shape = (28,28,1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train/=255
x_test/=255

model = Sequential()
model.add(Conv2D(28,kernel_size=(3,3),input_shape = input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation = tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10,activation = tf.nn.softmax))

model.compile(optimizer = 'adam' , loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train,train_labels,epochs =10)

test_loss, test_acc = model.evaluate(x_test,test_labels)
print(test_acc)


