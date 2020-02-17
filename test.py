import matplotlib
from matplotlib import pyplot as plt
matplotlib.interactive(True)

import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model

from PIL import Image

from ga import LineGA, PixelGA

input_images = "./data/camel.npy"
data = np.load(input_images)
data = data/255
img_w, img_h = 28, 28
data = np.reshape(data, [data.shape[0], img_w, img_h, 1])


batch_size = 128

# discriminator = load_model('trained.h5')
# print(discriminator.evaluate(data[:10], np.ones((10,))))
# initial_population = np.zeros((batch_size, img_h, img_w, img_h, img_w)).astype(np.bool)
# initial_population = np.random.rand(batch_size * 5, img_w, img_h, 1)
# generator = PixelGA(initial_population, 20, 50, 0.1)
# def calculate_fitness(population):
#     population = generator.in_pixel(population)
#     fitness = discriminator.predict(population.astype(np.float32)).flatten()
#     # fitness -= 0.5 * np.sum(population, axis=(1, 2, 3)) / population.shape[1] / population.shape[2]
#     return fitness
# generator.calculate_fitness = calculate_fitness

# for i in range(100):
#     generator.breed(10)

#     gen_imgs = generator.in_pixel(generator.select_best(16))
#     plt.figure(figsize=(5,5))
#     for k in range(15):
#         plt.subplot(4, 4, k+1)
#         plt.imshow(gen_imgs[k, :, :, 0], cmap='gray')
#         plt.axis('off')
#     plt.tight_layout()
#     plt.show()
#     plt.savefig('./images/camel_{}.png'.format(i+1))
#     plt.close('all')


# im = Image.open("cat.png")

# a = 1 - np.asarray(im).reshape(1, 28, 28, 1) / 255
# print(a)

# 
# print(model.predict(a))



plt.figure(figsize=(5,5))
for k in range(15):
    plt.subplot(4, 4, k+1)
    plt.imshow(data[k, :, :, 0], cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()
plt.savefig('./test.png')

