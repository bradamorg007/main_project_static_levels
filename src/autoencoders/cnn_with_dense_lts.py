import numpy as np
from keras import layers
from keras import backend as K
import keras
from keras.models import Model
from autoencoders.autoencoder_template import AutoEncoder
import os
import pickle as pkl


class CNN_DenseLatentSpace(AutoEncoder):

    def __init__(self, img_shape, latent_dimensions, batch_size, print_summary=False):
        super().__init__(img_shape, latent_space_dims=latent_dimensions, batch_size=batch_size)
        self.model_tag = 'CNN'
        self.print_summary = print_summary

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
        z_decoded = decoder(latent_vector)

        AE = Model(input_img, z_decoded)
        AE.compile(optimizer='rmsprop', loss='binary_crossentropy')

        encoder.compile(optimizer='rmsprop', loss='binary_crossentropy')
        decoder.compile(optimizer='rmsprop', loss='binary_crossentropy')

        if self.print_summary:
            encoder.summary()
            decoder.summary()
            AE.summary()

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

    skip_files = ['level_1', 'level_2',
                  'level_3', 'level_4', 'level_5', 'level_6', 'level_7', 'level_8', 'level_9']

    CNND = CNN_DenseLatentSpace(img_shape=(40, 40, 1), latent_dimensions=3, batch_size=1)
    CNND.data_prep_simple(directory_path='../AE_data/sim_test_main_capture/', skip_files=skip_files)

    CNND.y_train = [0]
    CNND.define_model()
    CNND.train(epochs=200)
    CNND.inspect_model(n=1, dim_reduction_model='pca', interval=1, labels=1, )
    CNND.save(name='CNND_main', save_type='weights_and_reconstruction_error')



"""   
    skip_files = ['level_1', 'level_2',
                  'level_3', 'level_4', 'level_5']

    CNND = CNN_DenseLatentSpace(img_shape=(40, 40, 1), latent_dimensions=1, batch_size=1)
    CNND.data_prep_simple(directory_path='../AE_data/test_report/', skip_files=[])

    split = 1
    y_train = []
    label = 0
    for i in range(len(CNND.x_train)):

        if i % split == 0:
            label += 1

        y_train.append(label)
    y_train = np.array(y_train)

    temp = CNND.x_train
    c =  [0] #[0,50,100,150,200,250,300]
    CNND.x_train = CNND.x_train[c]
    CNND.x_train = np.reshape(CNND.x_train[0], (1,) + CNND.x_train[0].shape)
    CNND.y_train = y_train
   #  # temp_x_data = CNND.x_train
    # CNND.x_train = CNND.x_train[0:split]
    #
    # CNND.y_train = y_train[0:split*3]
    #CNND.y_train = [0]
    CNND.define_model()
    CNND.train(epochs=200)
    #CNND.x_train = temp_x_data[0:split*3]
    CNND.x_train = temp
    CNND.inspect_model(n=11, dim_reduction_model='pca', interval=1, labels=11, )
    CNND.save(name='CNND_sim_test2', save_type='weights_and_reconstruction_error')
    
    """