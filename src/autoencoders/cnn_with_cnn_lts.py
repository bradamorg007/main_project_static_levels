import numpy as np
from keras import layers
from keras import backend as K
import keras
from keras.models import Model
from autoencoders.autoencoder_template import AutoEncoder
import os
import pickle as pkl



class CNN_ConvLatentSpace(AutoEncoder):

    def __init__(self, img_shape, latent_dimensions, batch_size):
        super().__init__(img_shape, latent_space_dims=latent_dimensions, batch_size=batch_size)
        self.model_type_flag = 'cnn_lts'


    def define_model(self):
        input_img = layers.Input(shape=self.img_shape)  # adapt this if using `channels_first` image data format

        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((5, 5), padding='same')(x)

        x = layers.Conv2D(self.latent_space_dims, (3, 3), activation='relu', padding='same')(x)
        latent_space = layers.MaxPooling2D((2, 2), padding='same')(x)

        encoder = Model(input_img, latent_space, name='Encoder')
        encoder.summary()

        # at this point the representation is (4, 4, 8) i.e. 128-dimensional

        decoder_inputs = layers.Input(shape=K.int_shape(latent_space)[1:])

        x = layers.Conv2D(self.latent_space_dims, (3, 3), activation='relu', padding='same')(decoder_inputs)
        x = layers.UpSampling2D((5, 5))(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)


        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)

        decoded_img = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        decoder = Model(decoder_inputs, decoded_img, name='decoder_model')
        decoder.summary()
        z_decoded = decoder(latent_space)

        AE = Model(input_img, z_decoded)
        AE.compile(optimizer='rmsprop', loss='binary_crossentropy')
        AE.summary()

        encoder.compile(optimizer='rmsprop', loss='binary_crossentropy')
        decoder.compile(optimizer='rmsprop', loss='binary_crossentropy')

        self.model = AE
        self.encoder = encoder
        self.decoder = decoder
        self.define_flag = True

    def load_weights(self, full_path):
        self.define_model()
        self.model.load_weights(os.path.join(full_path, 'weights_model.h5'))
        self.encoder.load_weights(os.path.join(full_path, 'weights_encoder_model.h5'))
        self.decoder.load_weights(os.path.join(full_path, 'weights_decoder_model.h5'))

        file_list = os.listdir(full_path)
        f_list = list(filter(lambda x: (x == 'reconstruction_error.pkl'), file_list))

        if len(f_list) != 0:
            self.reconstruction_error = pkl.load(open(os.path.join(full_path, 'reconstruction_error.pkl'), 'rb'))

        print('LOAD WEIGHTS COMPLETE')


if __name__ == "__main__":


    CNN = CNN_ConvLatentSpace(img_shape=(40, 40, 1), latent_dimensions=3, batch_size=128)
    CNN.data_prep(directory_path='../AE_data/data_seen_static/', skip_files=['.json'], data_index=0, label_index=1,
                  normalize=True, remove_blanks=True, data_type='train')
    CNN.map_labels_to_codes()

    CNN.define_model()
    CNN.train(epochs=50)
    CNN.inspect_model()
    CNN.save(name='CNN', save_type='weights')