from keras import layers
from keras import backend as K
import keras
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.models import load_model
from autoencoders.variational_autoencoder import VaritationalAutoEncoder
from autoencoders.variational_autoencoder_symmetric import VaritationalAutoEncoderSymmetric
from autoencoders.cnn_with_cnn_lts import CNN_ConvLatentSpace
from autoencoders.cnn_with_dense_lts import CNN_DenseLatentSpace

make_plot = False

def scatter_plot(pred, y_data, x_data, title='', cmap='Accent', interactive=False):

    if pred.shape[1] == 3:
        # Plot just how the latent space reacts to mix data
        fig = plt.figure(figsize=(6, 6))
        ax = Axes3D(fig)
        p = ax.scatter(pred[:, 0], pred[:, 1],pred[:, 1], c=y_data, cmap=plt.get_cmap(cmap))
        fig.colorbar(p,  fraction=0.060)
        plt.title(title)

        if interactive:
            annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                                bbox=dict(boxstyle="round", fc="w"),
                                arrowprops=dict(arrowstyle="->"))
            annot.set_visible(False)

            def update_annot(ind):

                pos = p.get_offsets()[ind["ind"][0]]
                annot.xy = pos
                text = "{}".format(" ".join(list(map(str, ind["ind"]))))

                annot.set_text(text)
                annot.get_bbox_patch().set_alpha(0.4)

            def hover(event):
                vis = annot.get_visible()
                if event.inaxes == ax:
                    cont, ind = p.contains(event)
                    if cont:
                        update_annot(ind)
                        annot.set_visible(True)

                        if event.key == 'x':
                            val = ind['ind']
                            print(val[0])
                            plt.figure()
                            plt.title(str(val[0]))
                            plt.imshow(x_data[val[0]].reshape(40, 40))
                            plt.gray()
                            plt.show()
                            print('KEY')


                        fig.canvas.draw_idle()
                    else:
                        if vis:
                            annot.set_visible(False)
                            fig.canvas.draw_idle()

                fig.canvas.mpl_connect("motion_notify_event", hover)
                fig.canvas.mpl_connect('key_press_event', hover)

        fig.show()

    elif pred.shape[1]== 2:
        fig = plt.figure(figsize=(6, 6))
        p = plt.scatter(pred[:, 0], pred[:, 1], c=y_data, cmap=plt.get_cmap(cmap))
        fig.colorbar(p,  fraction=0.060)
        plt.title(title)
        fig.show()



def plot_reconstructions(model, x_test, img_shape, title='', n=10):

    img_x = img_shape[0]
    img_y = img_shape[1]
    # Plot comparisons between original and decoded images with test data
    decoded_imgs = model.model.predict(x_test)
    plt.figure(figsize=(20, 4))

    # selecxt n random images to display

    selection = np.random.randint(decoded_imgs.shape[0], size=n)

    for i in range(n):
        # disp original

        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(img_x, img_y))
        plt.gray()

        ax = plt.subplot(2, n, i + n + 1)
        plt.imshow(decoded_imgs[i].reshape(img_x,img_y))
        plt.gray()

    plt.title(title)
    plt.show()

def combine_data(*args):
    return np.concatenate(args)


def shuffle(data_x, data_y):
    indices = np.arange(data_y.size)
    np.random.shuffle(indices)

    data_y = data_y[indices]
    data_x = data_x[indices]

    return data_x, data_y


def make_numeric(input, value):

    input[:] = value
    return input


# Define Model to be tested
MODEL = CNN_DenseLatentSpace(img_shape=(40, 40, 1), latent_dimensions=3, batch_size=128)
MODEL.load_weights(full_path='models/CNND')


# get data the model has trained on
MODEL.data_prep_simple(directory_path='../AE_data/test/', skip_files=['level_1', 'level_2', 'level_3', 'level_4', 'level_5'])
MODEL.y_train = [0]
x_train_seen, y_train_seen = [MODEL.x_train, MODEL.y_train]

#======================================================================================================================

# get data the 
MODEL.data_prep_simple(directory_path='../AE_data/test/', skip_files=['level_0','level_2', 'level_3', 'level_4', 'level_5'])
MODEL.y_train = [1]
x_train_unseen, y_train_unseen = [MODEL.x_train, MODEL.y_train]


MODEL.data_prep_simple(directory_path='../AE_data/test/', skip_files=['level_0', 'level_1','level_3', 'level_4', 'level_5'])
MODEL.y_train = [2]
x_train_unseen2, y_train_unseen2 = [MODEL.x_train, MODEL.y_train]


MODEL.data_prep_simple(directory_path='../AE_data/test/', skip_files=['level_0', 'level_1', 'level_2','level_4', 'level_5'])
MODEL.y_train = [3]
x_train_unseen3, y_train_unseen3 = [MODEL.x_train, MODEL.y_train]


MODEL.data_prep_simple(directory_path='../AE_data/test/', skip_files=['level_0', 'level_1', 'level_2', 'level_3', 'level_5'])
MODEL.y_train = [4]
x_train_unseen4, y_train_unseen4 = [MODEL.x_train, MODEL.y_train]


MODEL.data_prep_simple(directory_path='../AE_data/test/', skip_files=['level_0', 'level_1', 'level_2', 'level_3', 'level_4'])
MODEL.y_train = [5]
x_train_unseen5, y_train_unseen5 = [MODEL.x_train, MODEL.y_train]



# y_train_seen    = make_numeric(y_train_seen,    0)
# y_train_unseen  = make_numeric(y_train_unseen,  1)
# y_train_unseen2 = make_numeric(y_train_unseen2, 2)
# y_train_unseen3 = make_numeric(y_train_unseen3, 3)
# y_train_unseen4 = make_numeric(y_train_unseen4, 4)
# y_train_unseen5 = make_numeric(y_train_unseen5, 5)


x_train_mix = combine_data(x_train_seen, x_train_unseen, x_train_unseen2,
                           x_train_unseen3,x_train_unseen4, x_train_unseen5 )

y_train_mix = combine_data(y_train_seen, y_train_unseen, y_train_unseen2,
                           y_train_unseen3,y_train_unseen4, y_train_unseen5)

x_train_mix, y_train_mix = shuffle(x_train_mix, y_train_mix)

pred_unseen = MODEL.predict(MODEL.encoder, x_train_unseen, 16, 'pca', 3)
pred_seen   = MODEL.predict(MODEL.encoder, x_train_seen, 16, 'pca', 3)
pred_mix    = MODEL.predict(MODEL.encoder, x_train_mix, 16, 'pca', 3)

scatter_plot(pred=pred_mix, y_data=y_train_mix, x_data=x_train_mix,
             title='Latent Space THROUGH PREDICTION: Seen & unseen data')

scatter_plot(pred=pred_seen, y_data=y_train_seen, x_data=x_train_seen,
             title='Latent Space: Seen data')

scatter_plot(pred=pred_unseen, y_data=y_train_unseen, x_data=x_train_unseen,
             title='Latent Space: Unseen data')

img_shape = [40, 40]

plot_reconstructions(model=MODEL, x_test=x_train_mix, img_shape=img_shape, title='mix', n=6)
plot_reconstructions(model=MODEL, x_test=x_train_unseen, img_shape=img_shape, title='unseen',n=6)
plot_reconstructions(model=MODEL, x_test=x_train_seen, img_shape=img_shape, title='seen',n=6)

seen_RE = MODEL.model.evaluate(x_train_seen, x_train_seen, batch_size=16)
unseen_RE =  MODEL.model.evaluate(x_train_unseen, x_train_unseen, batch_size=16)


plt.figure(figsize=(6, 6))
plt.bar(x=np.arange(len([seen_RE, unseen_RE])), height=[seen_RE, unseen_RE], tick_label=['seen_RE', 'unseen_RE'])
plt.show()