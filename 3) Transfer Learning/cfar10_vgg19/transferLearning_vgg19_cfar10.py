from keras.applications.vgg19 import VGG19
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Flatten
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import cv2
import numpy as np

#%%
(x_train, y_train),(x_test, y_test) = cifar10.load_data()
print("x_train shape",x_train.shape)
print("train sample:",x_train.shape[0])

numberOfClass = 10

y_train = to_categorical(y_train, numberOfClass)
y_test = to_categorical(y_test, numberOfClass)

input_shape = x_train.shape[1:]

#%% visualize
plt.imshow(x_train[5511].astype(np.uint8))
plt.axis("off")
plt.show()

# %% increase dimension
def resize_img(img):
    numberOfImage = img.shape[0]
    new_array = np.zeros((numberOfImage, 48,48,3))
    for i in range(numberOfImage):
        new_array[i] = cv2.resize(img[i,:,:,:],(48,48))
    return new_array

x_train = resize_img(x_train)
x_test = resize_img(x_test)
print("increased dim x_train: ",x_train.shape)

plt.figure()
plt.imshow(x_train[5511].astype(np.uint8))
plt.axis("off")
plt.show()

#%% vgg19

vgg = VGG19(include_top = False, weights = "imagenet", input_shape = (48,48,3))

print(vgg.summary())

vgg_layer_list = vgg.layers
print(vgg_layer_list)

model = Sequential()
for layer in vgg_layer_list:
    model.add(layer)
    
print(model.summary())

for layer in model.layers:
    layer.trainable = False

# fully con layers
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(numberOfClass, activation= "softmax"))

print(model.summary())


model.compile(loss = "categorical_crossentropy",
              optimizer = "rmsprop",
              metrics = ["accuracy"])

hist = model.fit(x_train, y_train, validation_split = 0.2, epochs = 5, batch_size = 1000)

#%%  model save
model.save_weights("example.h5")

#%%
plt.plot(hist.history["loss"], label = "train loss")
plt.plot(hist.history["val_loss"], label = "val loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(hist.history["acc"], label = "train acc")
plt.plot(hist.history["val_acc"], label = "val acc")
plt.legend()
plt.show()

#%% load
import json, codecs
with codecs.open("transfer_learning_vgg19_cfar10.json","r",encoding = "utf-8") as f:
    n = json.loads(f.read())

plt.plot(n["acc"], label = "train acc")
plt.plot(n["val_acc"], label = "val acc")
plt.legend()
plt.show()


#%% save
with open('transfer_learning_vgg19_cfar10.json', 'w') as f:
    json.dump(hist.history, f)




































