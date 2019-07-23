import numpy as np
from keras import layers
from keras import backend as K
import keras
from keras.models import Model
from autoencoders.autoencoder_template import AutoEncoder
import os



class FullyConnectedAutoEncoder(AutoEncoder):

    def __init__(self, img_shape, latent_dimensions, batch_size):
        super().__init__(img_shape, latent_space_dims=latent_dimensions, batch_size=batch_size)


    def define_model(self):


        input_img = layers.Input(shape=self.img_shape)

        # ENCODER ==================================================================================================
        shape_before_flattening = K.int_shape(input_img)
        x = layers.Flatten()(input_img)
        x = layers.Dense(units=1600, activation='relu')(x)
        x = layers.Dense(units=1200, activation='relu')(x)
        x = layers.Dense(units=800, activation='relu')(x)
        x = layers.Dense(units=400, activation='relu')(x)
        x = layers.Dense(units=100, activation='relu')(x)
        x = layers.Dense(units=50, activation='relu')(x)
        x = layers.Dense(units=25, activation='relu')(x)



        latent_vector = layers.Dense(units=self.latent_space_dims, name='Latent_space',
                                     activity_regularizer=layers.regularizers.l1(10e-5))(x)

        encoder = Model(input_img, latent_vector, name='Encoder')
        encoder.summary()

        # DECODER =========================================================================================
        decoder_inputs = layers.Input(shape=K.int_shape(latent_vector))

        x = layers.Dense(units=25, activation='relu')(decoder_inputs)
        x = layers.Dense(units=50, activation='relu')(x)
        x = layers.Dense(units=100, activation='relu')(x)
        x = layers.Dense(units=400, activation='relu')(x)
        x = layers.Dense(units=800, activation='relu')(x)
        x = layers.Dense(units=1200, activation='relu')(x)
        x = layers.Dense(units=1600, activation='sigmoid')(x)
        decoded_img = layers.Reshape(target_shape=shape_before_flattening[1:])(x)

        # decoded_img = layers.Conv2D(filters=1, kernel_size=(3, 3), padding='same',
        #                             activation='sigmoid', name='decoded_img')(d)

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

        print('LOAD WEIGHTS COMPLETE')


if __name__ == '__main__':
    FCAE = FullyConnectedAutoEncoder(img_shape=(40, 40, 1), latent_dimensions=3, batch_size=128)
    FCAE.data_prep(directory_path='../AE_data/data_seen_static/', skip_files=['.json'], data_index=0, label_index=1,
                   normalize=True, remove_blanks=True, data_type='train')

    FCAE.map_labels_to_codes()
    FCAE.define_model()
    FCAE.train(epochs=20)
    FCAE.inspect_model()
    FCAE.save(name='FCAE', save_type='weights')