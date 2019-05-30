# https://blog.keras.io/building-autoencoders-in-keras.html
from keras.models import Model
from keras.layers import Input, Dense
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import json, codecs
import warnings
warnings.filterwarnings("ignore")

(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape((len(x_train), x_train.shape[1:][0]*x_train.shape[1:][1]))
x_test = x_test.reshape((len(x_test), x_test.shape[1:][0]*x_test.shape[1:][1]))

plt.imshow(x_train[4000].reshape(28,28))
plt.axis("off")
plt.show()

#%%

input_img = Input(shape = (784,))

encoded = Dense(32, activation="relu")(input_img)

encoded = Dense(16, activation="relu")(encoded)

decoded = Dense(32, activation="relu")(encoded)

decoded = Dense(784, activation="sigmoid")(decoded)

autoencoder = Model(input_img,decoded)

autoencoder.compile(optimizer="rmsprop",loss="binary_crossentropy")

hist = autoencoder.fit(x_train,
                       x_train,
                       epochs=200,
                       batch_size=256,
                       shuffle=True,
                       validation_data = (x_train,x_train))

#%% save model
autoencoder.save_weights("autoencoder_model.h5")

#%% evaluation
print(hist.history.keys())

plt.plot(hist.history["loss"],label = "Train loss")
plt.plot(hist.history["val_loss"],label = "Val loss")

plt.legend()
plt.show()

# %% save hist
with open("autoencoders_hist.json","w") as f:
    json.dump(hist.history,f)


# %% load history
with codecs.open("autoencoders_hist.json","r", encoding="utf-8")  as f:
    n = json.loads(f.read())
#%% 
print(n.keys())
plt.plot(n["loss"],label = "Train loss")
plt.plot(n["val_loss"],label = "Val loss")

#%% 
encoder = Model(input_img,encoded)
encoded_img = encoder.predict(x_test)

plt.imshow(x_test[1500].reshape(28,28))
plt.axis("off")
plt.show()

plt.figure()
plt.imshow(encoded_img[1500].reshape(4,4))
plt.axis("off")
plt.show()

decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.axis("off")

    # display reconstruction
    ax = plt.subplot(2, n, i + n+1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.axis("off")
plt.show()

















