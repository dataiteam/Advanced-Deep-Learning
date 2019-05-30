from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
from glob import glob

train_path = "fruits-360/Training/"
test_path = "fruits-360/Test/"

img = load_img(train_path + "Avocado/0_100.jpg")
plt.imshow(img)
plt.axis("off")
plt.show()

x = img_to_array(img)
print(x.shape)

numberOfClass = len(glob(train_path+"/*"))

vgg = VGG16()

print(vgg.summary())
print(type(vgg))

vgg_layer_list = vgg.layers
print(vgg_layer_list)

model = Sequential()
for i in range(len(vgg_layer_list)-1):
    model.add(vgg_layer_list[i])

print(model.summary())

for layers in model.layers:
    layers.trainable = False

model.add(Dense(numberOfClass, activation="softmax"))

print(model.summary())

model.compile(loss = "categorical_crossentropy",
              optimizer = "rmsprop",
              metrics = ["accuracy"])

# train  
train_data = ImageDataGenerator().flow_from_directory(train_path,target_size = (224,224))
test_data = ImageDataGenerator().flow_from_directory(test_path,target_size = (224,224))

batch_size = 32

hist = model.fit_generator(train_data,
                           steps_per_epoch=1600//batch_size,
                           epochs= 25,
                           validation_data=test_data,
                           validation_steps= 800//batch_size)

#%%
model.save_weights("deneme.h5")

#%% evaluation
print(hist.history.keys())
plt.plot(hist.history["loss"],label = "training loss")
plt.plot(hist.history["val_loss"],label = "validation loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(hist.history["acc"],label = "training acc")
plt.plot(hist.history["val_acc"],label = "validation acc")
plt.legend()
plt.show()

#%% save history
import json, codecs
with open("deneme.json","w") as f:
    json.dump(hist.history,f)
    
#%% load history
with codecs.open("transfer_learning_fruit_hist.json","r",encoding = "utf-8") as f:
    n = json.loads(f.read())

plt.plot(n["loss"],label = "training loss")
plt.plot(n["val_loss"],label = "validation loss")
plt.legend()
plt.show()
plt.figure()
plt.plot(n["acc"],label = "training acc")
plt.plot(n["val_acc"],label = "validation acc")
plt.legend()
plt.show()





























