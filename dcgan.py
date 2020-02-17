"""Build a traditional GAN as a comparison""" 

import matplotlib
from matplotlib import pyplot as plt


import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop, Adam

noise_dim = 100

def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(noise_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))

    model.add(layers.Conv2DTranspose(128, 5, strides=1, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, 5, strides=2, padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, 5, strides=2, padding='same', use_bias=False, activation='sigmoid'))
    return model


def build_discriminator():
    depth = 64
    p = 0.4
    model = tf.keras.Sequential()
    
    model.add(layers.Conv2D(depth*1, 5, strides=2, padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.Dropout(p))
    
    model.add(layers.Conv2D(depth*2, 5, strides=2, padding='same', activation='relu'))
    model.add(layers.Dropout(p))
    
    model.add(layers.Conv2D(depth*4, 5, strides=2, padding='same', activation='relu'))
    model.add(layers.Dropout(p))
    
    model.add(layers.Conv2D(depth*8, 5, strides=1, padding='same', activation='relu'))
    model.add(layers.Dropout(p))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()
    return model

def build_combination(generator, discriminator):
    z = layers.Input(shape=(noise_dim,))
    img = generator(z)
    discriminator.trainable = False
    validity = discriminator(img)
    combination = tf.keras.Model(inputs=z, outputs=validity)
    combination.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return combination

epochs = 1000
batch_size = 128
input_images = "./data/camel.npy"
data = np.load(input_images)
data = data/255
img_w, img_h = 28, 28
data = np.reshape(data, [data.shape[0], img_w, img_h, 1])


discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
generator = build_generator()
gan = build_combination(generator, discriminator)


def sample_images(generator, epoch):
    r, c = 3, 3
    noise = np.random.normal(0, 1, (r * c, noise_dim))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("images/%d.png" % epoch)
    plt.close()


for epoch in range(epochs):
    real_data = np.array(data[np.random.choice(len(data), batch_size, replace=False)])
    noise = np.random.normal(0, 1, (batch_size, noise_dim))
    fake_data = generator.predict(noise)

    d_loss = discriminator.train_on_batch(real_data, np.ones(batch_size))
    d_loss += discriminator.train_on_batch(fake_data, np.zeros(batch_size))
    d_loss *= 0.5


    noise = np.random.normal(0, 1, (batch_size, noise_dim)) 

    g_loss = gan.train_on_batch(noise, np.ones(batch_size))
    print('dloss {}, gloss {}, epoch {}'.format(d_loss, g_loss, epoch))

    if epoch % 10 == 0:
        sample_images(generator, epoch)
