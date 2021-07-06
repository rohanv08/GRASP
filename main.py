import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Input, Concatenate, Conv2D, LeakyReLU, BatchNormalization, Activation, \
    Conv2DTranspose, Dropout
from tensorflow.keras import models, Model, initializers
from tensorflow.keras.optimizers import Adam
from os import listdir
from matplotlib import pyplot
import cv2


def load_images(path_dir, size=(256, 512)):
    src_list, tar_list = list(), list()
    for filename in listdir(path_dir):
        if '.PNG' in filename:
            pixels = load_img(path_dir + filename, target_size=size)
            pixels = img_to_array(pixels)
            src_list.append(pixels[:, :256])
            tar_list.append(pixels[:, 256:])
    return [np.asarray(src_list), np.asarray(tar_list)]


def generate_real_samples(dataset, n_samples, patch_shape):
    trainA, trainB = dataset
    ix = np.random.randint(0, trainA.shape[0], n_samples)
    X1, X2 = trainA[ix], trainB[ix]
    y = np.ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y


def generate_fake_samples(g_model, samples, patch_shape):
    X = g_model.predict(samples)
    y = np.zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


def discriminator(shape=(256, 256, 3)):
    init = initializers.RandomNormal(stddev=0.02)
    in_img = Input(shape=shape)
    tar_img = Input(shape=shape)
    merged_img = Concatenate()([in_img, tar_img])
    layer = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged_img)
    layer = LeakyReLU(alpha=0.2)(layer)

    layer = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer)
    layer = BatchNormalization()(layer)
    layer = LeakyReLU(alpha=0.2)(layer)

    layer = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer)
    layer = BatchNormalization()(layer)
    layer = LeakyReLU(alpha=0.2)(layer)

    layer = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer)
    layer = BatchNormalization()(layer)
    layer = LeakyReLU(alpha=0.2)(layer)

    layer = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(layer)
    layer = BatchNormalization()(layer)
    layer = LeakyReLU(alpha=0.2)(layer)

    layer = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(layer)
    result = Activation('sigmoid')(layer)

    model = Model([in_img, tar_img], result)
    opt = Adam(learning_rate=0.002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model


def encoder(layer, no_filter, batch=True):
    init = initializers.RandomNormal(stddev=0.02)
    layer_encode = Conv2D(no_filter, (4, 4), strides=(2, 2), padding='same',
                          kernel_initializer=init)(layer)
    if batch:
        layer_encode = BatchNormalization()(layer_encode)
    layer_encode = LeakyReLU(alpha=0.2)(layer_encode)
    return layer_encode


def decoder(layer, skip, no_filter, dropout=True):
    init = initializers.RandomNormal(stddev=0.02)
    layer_decode = Conv2DTranspose(no_filter, (4, 4), strides=(2, 2), padding='same',
                                   kernel_initializer=init)(layer)
    layer_decode = BatchNormalization()(layer_decode)
    if dropout:
        layer_decode = Dropout(0.5)(layer_decode)
    layer_decode = Concatenate()([layer_decode, skip])
    layer_decode = Activation('relu')(layer_decode)
    return layer_decode


def generator(image_shape=(256, 256, 3)):
    in_image = Input(shape=image_shape)
    init = initializers.RandomNormal(stddev=0.02)
    e1 = encoder(in_image, 64, batch=False)
    e2 = encoder(e1, 128)
    e3 = encoder(e2, 256)
    e4 = encoder(e3, 512)
    e5 = encoder(e4, 512)
    e6 = encoder(e5, 512)
    e7 = encoder(e6, 512)

    b = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(e7)
    b = Activation('relu')(b)

    d1 = decoder(b, e7, 512)
    d2 = decoder(d1, e6, 512)
    d3 = decoder(d2, e5, 512)
    d4 = decoder(d3, e4, 512, dropout=False)
    d5 = decoder(d4, e3, 256, dropout=False)
    d6 = decoder(d5, e2, 128, dropout=False)
    d7 = decoder(d6, e1, 64, dropout=False)

    g = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same')(d7)
    out_image = Activation('tanh')(g)
    model = Model(in_image, out_image)
    return model


def gan(gen_model, dis_model, image_shape):
    for layer in dis_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    input_src = Input(shape=image_shape)
    gen_out = gen_model(input_src)
    dis_out = dis_model([input_src, gen_out])
    model = Model(input_src, [dis_out, gen_out])
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])
    return model


def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
    n_patch = d_model.output_shape[1]
    trainA, trainB = dataset
    bat_per_epo = int(len(trainA) / n_batch)
    print(bat_per_epo)
    n_steps = bat_per_epo * n_epochs
    # manually enumerate epochs
    for i in range(n_steps):
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i + 1, d_loss1, d_loss2, g_loss))
        # summarize model performance
        if (i + 1) % (bat_per_epo) == 0:
            summarize_performance(i, g_model, dataset)


def summarize_performance(step, g_model, dataset, n_samples=3):
    # select a sample of input images
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    # scale all pixels from [-1,1] to [0,1]
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0
    # plot real source images
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realA[i])
    # plot generated target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(X_fakeB[i])
    # plot real target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples * 2 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realB[i])
    # save plot to file
    filename1 = 'plot_%06d.png' % (step + 1)
    pyplot.savefig('plots/' + filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'model.h5'
    g_model.save('models/' + filename2)
    print('>Saved: %s and %s' % (filename1, filename2))


def load_real_samples(filename):
    data = np.load(filename)
    print(data)
    # unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]



def train_main(load):
    dataset = load_real_samples(load)
    print('Loaded', dataset[0].shape, dataset[1].shape)
    image_shape = dataset[0].shape[1:]
    d_model = discriminator(image_shape)
    g_model = generator(image_shape)
    gan_model = gan(g_model, d_model, image_shape)
    train(d_model, g_model, gan_model, dataset)
    g_model.save('trained_model.h5')
    print("saved")


def create_npz(path, filename):
    print('started')
    [src_images, tar_images] = load_images(path)
    np.savez_compressed(filename, src_images, tar_images)
    print('Saved dataset: ', filename)
    print(src_images.shape, tar_images.shape)


def concatenate (original, annotated, concatenated):
    for filename in listdir(original):
        if '.PNG' in filename:
            original_im = cv2.imread(original + filename)
            annotated_im = cv2.imread(annotated + filename)
            concat = cv2.hconcat([original_im, annotated_im])
            cv2.imwrite(concatenated + filename, concat)
    print('done')

def load_and_validate(model_path, dataset_path):
    g_model = models.load_model(model_path)
    optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    g_model.compile(optimizer=optimizer,
                  loss=['binary_crossentropy', 'mae'],
                  metrics=['accuracy'])
    dataset = load_real_samples(dataset_path)
    print('Loaded', dataset[0].shape, dataset[1].shape)
    [src, tar] = dataset
    results = g_model.evaluate(src, tar)
    print(results)

def load_and_test(model_path, dataset_path):
    g_model = models.load_model(model_path)
    dataset = load_real_samples(dataset_path)
    print('Loaded', dataset[0].shape, dataset[1].shape)
    [src, tar] = dataset
    sample = src[:1000]
    print(sample.shape)
    predicted = g_model.predict(sample)
    #predicted = (predicted + 1) / 2.0
    #tar = (tar + 1) / 2.0
    print(len(predicted))
    for i in range(len(predicted)):
        pyplot.subplot(1, 2, 1)
        pyplot.imshow(predicted[i])
        pyplot.subplot(1, 2, 2)
        pyplot.imshow(tar[i])
        pyplot.axis('off')
        filename = str(i) + '.png'
        pyplot.savefig('plots/' + filename)
    print('done')







#concatenate('data/1.1/', 'data/1.1/annotated/', 'data/1.1/concatenated/')
#train_main('1.2.npz')
#create_npz('data/1.1/concatenated/', '1.1.npz')
load_and_test('models/model.h5', '1.1.npz')


