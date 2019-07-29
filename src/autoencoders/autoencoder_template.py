import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from old_code.file_extractor import FileExtractor
from sklearn.manifold import TSNE
import sklearn.decomposition as decomposition
import time
import seaborn as sns
import pickle as pkl

class AutoEncoder:

    def __init__(self, img_shape, latent_space_dims, batch_size):

       self.img_shape = img_shape
       self.latent_space_dims = latent_space_dims
       self.encoder = None
       self.latent_space = None
       self.decoder = None
       self.model = None

       self.batch_size = batch_size
       self.x_train = None
       self.y_train = None
       self.x_test = None
       self.y_test = None
       self.blanks_detected = [0, []]
       self.history = None

       self.label_table_train = None
       self.label_keys_train = None
       self.label_table_test = None
       self.label_keys_test = None

       self.define_flag = False
       self.data_flag = False
       self.train_flag = False
       self.label_config_flag = True
       self.model_type_flag = None



    def data_prep_simple(self, directory_path, skip_files):
        data = FileExtractor.extract(directory_path=directory_path, skip_files=skip_files)

        x = []

        for sample in data:
            x.append(np.reshape(sample.astype('float32'), newshape=sample.shape + (1,)))

        x = np.array(x)
        self.x_train = x
        self.data_flag = True

    def data_prep(self, directory_path, skip_files, data_index,
                  label_index, normalize, remove_blanks, data_type):

        self.blanks_detected = [0, []]
        data = FileExtractor.extract(directory_path=directory_path, skip_files=skip_files)

        x = []
        y = []

        for i in range(len(data)):
            sample = data[i][data_index]
            label = data[i][label_index]


            if remove_blanks:

                if sample.min() != sample.max():
                    if normalize:
                        sample = sample / 255

                    x.append(np.reshape(sample.astype('float32'), newshape=sample.shape + (1,)))
                    y.append(label)
                else:
                    self.blanks_detected[0] += 1
                    self.blanks_detected[1].append(i)



        x = np.array(x)
        y = np.array(y)

        if data_type == 'train':
            self.x_train = x
            self.y_train = y
            self.data_flag = True
            self.label_config('train')

        elif data_type == 'test':
            self.x_test = x
            self.y_test = y
            self.data_flag = True
            self.label_config('test')

        else:
            raise ValueError('ERROR data prep: Please select a valid data type, either train or test data')


    def label_config(self, data_type):

        y = None
        if data_type == 'train':
            y = self.y_train

        elif data_type == 'test':
            y = self.y_test
        else:
            raise ValueError('ERROR data prep: Please select a valid data type, either train or test data')


        if self.data_flag:
             label_table = AutoEncoder.count_unquie(y)

             if data_type == 'train':
                self.label_table_train = label_table
                self.label_keys_train = label_table.keys()

             elif data_type == 'test':
                 self.label_table_test = label_table
                 self.label_keys_test = label_table.keys()

        else:
            raise ValueError("ERROR Label Config: Please use the data prep function before using this function")

    def map_labels_to_codes(self):

        new_y_train = np.zeros(shape=self.y_train.shape)
        for i in range(len(self.y_train)):

            code = self.label_table_train.get(self.y_train[i]).get('code')
            new_y_train[i] = code

        self.y_train = new_y_train


    def filter(self, keep_labels, data_type):

        label_table = None
        if data_type == 'train':
            label_table = self.label_table_train

        elif data_type == 'test':
            label_table = self.label_table_test
        else:
            raise ValueError('ERROR data prep: Please select a valid data type, either train or test data')


        if self.label_config_flag:
            if isinstance(keep_labels, list):

                keep_labels_indexes = []
                for label in keep_labels:

                    lookup = label_table.get(label)

                    if lookup is not None:
                       keep_labels_indexes = keep_labels_indexes + lookup.get('indices')
                    else:
                        raise ValueError('ERROR filter: Element in keep labels list does not exist in the label _table')

                if data_type == 'train':
                    a = self.x_train[keep_labels_indexes]
                    self.x_train = self.x_train[keep_labels_indexes]
                    self.y_train = self.y_train[keep_labels_indexes]
                elif data_type == 'test':
                    self.x_test = self.x_test[keep_labels_indexes]
                    self.y_test = self.y_test[keep_labels_indexes]


            else:
                raise ValueError('ERROR filter: keep_labels must of type list')
        else:
            raise ValueError("ERROR filter: Please use the label config function before using this function")


    def train(self, epochs):

        if self.define_flag and self.data_flag:
            # =======================================================================================================================
            history = self.model.fit(x=self.x_train, y=self.x_train,
                                   shuffle=True, epochs=epochs, batch_size=self.batch_size,
                                   validation_split=0, verbose=2)

            self.history = history
            self.train_flag = True

        else:
            raise ValueError('ERROR: THE MODEL AND THE DATA MUST BE DEFINED BEFORE TRAIN CAN BE CALLED')


    def predict(self, model, input, batch_size, dim_reduction_model, dimensions, dim_reduce=True):

        pred = model.predict(input, batch_size=batch_size)

        if isinstance(pred, list) and len(pred) > 1:
            # if variational AE used just use the mean value outputs not the stdevs
            pred = pred[0]

        if self.model_type_flag == 'cnn_lts':
            pred = np.reshape(pred, newshape=(pred.shape[0], pred.shape[1] * pred.shape[2] * pred.shape[3]))

        if pred.shape[1] > 3 and dim_reduce == True:

            if dim_reduction_model == 'tsne':
                # perform tsne
                pred = self.tsne(pred, dimensions=dimensions)

            elif dim_reduction_model == 'pca':
                pred = self.pca(pred, dimensions=dimensions)


        return pred


    def tsne(self, input, dimensions):
        time_start = time.time()
        tsne = TSNE(n_components=dimensions, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(input)
        print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
        return tsne_results


    def pca(self, input, dimensions):
        time_start = time.time()
        pca = decomposition.PCA(n_components=dimensions)
        pca.fit(input)
        input = pca.transform(input)
        print('PCA done! Time elapsed: {} seconds'.format(time.time() - time_start))
        return input

    def show_label_codes(self):

        if self.label_table_train is not None:
            for key in self.label_table_train:
                print('Label: %s Code: %s' % (key, self.label_table_train.get(key).get('code')))


    def save(self, name, save_type):

        folder = 'models/'

        if os.path.isdir(folder) == False:
            os.mkdir(folder)

        if os.path.isdir(os.path.join(folder, name)) == False:
            os.mkdir(os.path.join(folder, name))


        if save_type == 'model':

            self.model.save(os.path.join(folder, name, 'VAE_full_model.h5'))
            self.encoder.save(os.path.join(folder, name, 'encoder_model.h5'))
            self.latent_space.save(os.path.join(folder, name, 'latent_space_model.h5'))
            self.decoder.save(os.path.join(folder, name, 'decoder_model.h5'))

            self.model.save_weights(os.path.join(folder, name, 'weights_model.h5'))
            self.encoder.save_weights(os.path.join(folder, name, 'weights_encoder_model.h5'))
            self.decoder.save_weights(os.path.join(folder, name, 'weights_decoder_model.h5'))
            print('SAVE MODEL COMPLETE')

        elif save_type == 'weights':
            self.model.save_weights(os.path.join(folder, name, 'weights_model.h5'))
            self.encoder.save_weights(os.path.join(folder, name, 'weights_encoder_model.h5'))
            self.decoder.save_weights(os.path.join(folder, name, 'weights_decoder_model.h5'))
            print('SAVE WEIGHTS COMPLETE')

        elif save_type == 'weights_and_reconstruction_error':

            if self.x_test is not None:
              RE = self.model.evaluate(self.x_test, self.x_test)

            elif self.x_train is not None:
              RE = self.model.evaluate(self.x_train, self.x_train)

            else:
                raise ValueError('ERROR AUTOENCODER: No Data is available to perform an evaluation')


            self.model.save_weights(os.path.join(folder, name, 'weights_model.h5'))
            self.encoder.save_weights(os.path.join(folder, name, 'weights_encoder_model.h5'))
            self.decoder.save_weights(os.path.join(folder, name, 'weights_decoder_model.h5'))
            filename = open(os.path.join(folder, name,'reconstruction_error.pkl'), 'wb')
            pkl.dump(RE, filename)
            filename.close()
            print('SAVE WEIGHTS COMPLETE: Reconstrunction Error = %s' % RE)


    def inspect_model(self, dim_reduction_model='tsne', dimensions=2, n=4, plot_rand=False):
        # PLOTTING & METRICS===================================================================================================

        # plot the latent space of the VAE
        #encoder = Model(input_img, [z_mean, z_log_var, latent_space], name='encoder')
        self.show_label_codes()
        pred = self.predict(model=self.encoder, input=self.x_train,
                            batch_size=self.batch_size, dim_reduction_model=dim_reduction_model,
                            dimensions=dimensions)

        c = None
        if self.label_keys_train is None:
            c = len(self.y_train)
        else:
            c = len(self.label_keys_train)

        if pred.shape[1] == 2:
            plt.figure(figsize=(16, 10))
            sns.scatterplot(
                x=pred[:, 0], y=pred[:, 1],
                hue=self.y_train,
                palette=sns.color_palette("hls", c),
                legend="full",
                alpha=0.3
            )

        elif pred.shape[1] == 3:
            fig = plt.figure(figsize=(6, 6))
            ax = Axes3D(fig)
            p = ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], c=self.y_train)
            fig.colorbar(p)
            fig.show()


        # Plot comparisons between original and decoded images with test data
        decoded_imgs = self.model.predict(self.x_train)
        if plot_rand:
            image_samples = np.random.randint(0, len(self.x_train), size=n)
        else:
            image_samples = np.arange(n)
        plt.figure(figsize=(20, 4))

        for i in range(n):
            # disp original

            ax = plt.subplot(2, n, i + 1)
            plt.imshow(self.x_train[image_samples[i]].reshape(40, 40))
            plt.gray()

            ax = plt.subplot(2, n, i + n + 1)
            plt.imshow(decoded_imgs[image_samples[i]].reshape(40, 40))
            plt.gray()

        plt.show()

        # Plot Traning and validation reconstruction error
        loss = self.history.history['loss']
        val_loss = self.history.history.get('val_loss')

        epochs = range(1, len(loss) + 1)

        plt.figure()

        plt.plot(epochs, loss, 'g', label='training loss')
        if val_loss is not None:
            plt.plot(epochs, val_loss, 'r', label='validation loss')
        plt.title('Training and Validation Reconstruction Error')
        plt.legend()

        plt.show()

    @staticmethod
    def count_unquie(input):
        table = {}

        for i, rows in enumerate(input):

            # try to retrieve a label key, if it exists append the freq count and add i to indices

            lookup = table.get(rows)

            if lookup is not None:

                lookup['freq'] += 1
                lookup['indices'].append(i)

            else:
                # if it doesnt exist in the dict then make a new entry
                table[rows] = {'freq': 1, 'indices': [i]}

        for i, key in enumerate(table.keys()):
            table[key]['code'] = i

        return table



if __name__ == '__main__':

    pass
    #VAE.save(name='trained_digits_8', save_type='weights')
    #
    # VAE.load_weights(full_path='..\mnist_AE\models/test')
    # print('Seen Reconstruction Error: %s ' % VAE.VAE.evaluate(VAE.x_test, VAE.x_test, batch_size=16))
    #
    # VAE.data_prep(keep_labels=[7])
    # print('Unseen Reconstruction Error: %s ' % VAE.VAE.evaluate(VAE.x_test, VAE.x_test, batch_size=16))
