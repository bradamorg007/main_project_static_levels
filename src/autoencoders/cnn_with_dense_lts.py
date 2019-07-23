import numpy as np
from keras import layers
from keras import backend as K
import keras
from keras.models import Model
from autoencoders.autoencoder_template import AutoEncoder
import os
import pickle as pkl


class CNN_DenseLatentSpace(AutoEncoder):

    def __init__(self, img_shape, latent_dimensions, batch_size):
        super().__init__(img_shape, latent_space_dims=latent_dimensions, batch_size=batch_size)
        self.model_tag = 'CNN'

    def define_model(self):
        # INPUT LAYER

        input_img = layers.Input(shape=self.img_shape)

        # ENCODER ==================================================================================================

        x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_img)
        x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', strides=2)(x)
        x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)

        shape_before_flattening = K.int_shape(x)
        x = layers.Flatten()(x)
        x = layers.Dense(units=50, activation='relu')(x)

        latent_vector = layers.Dense(units=self.latent_space_dims, name='Latent_space',
                                     activity_regularizer=layers.regularizers.l1(10e-5))(x)
        encoder = Model(input_img, latent_vector, name='Encoder')
        encoder.summary()

        # DECODER =========================================================================================
        decoder_inputs = layers.Input(shape=K.int_shape(latent_vector)[1:])

        d = layers.Dense(units=50, activation='relu')(decoder_inputs)
        d = layers.Dense(units=np.prod(shape_before_flattening[1:]), activation='relu')(d)
        d = layers.Reshape(target_shape=shape_before_flattening[1:])(d)

        d = layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(d)
        d = layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(d)
        d = layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same', activation='relu', strides=2)(d)
        d = layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(d)

        decoded_img = layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same',
                                    activation='sigmoid', name='decoded_img')(d)

        decoder = Model(decoder_inputs, decoded_img, name='decoder_model')
        decoder.summary()
        z_decoded = decoder(latent_vector)

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


if __name__ == '__main__':

    CNND = CNN_DenseLatentSpace(img_shape=(40, 40, 1), latent_dimensions=15, batch_size=64)
    CNND.data_prep(directory_path='../AE_data/data_1_static/', skip_files=['.json'], data_index=0, label_index=1,
                   normalize=True, remove_blanks=True, data_type='train')

    CNND.map_labels_to_codes()
    CNND.define_model()
    CNND.train(epochs=20)
    CNND.inspect_model()
    CNND.save(name='CNND', save_type='weights_and_reconstruction_error')
