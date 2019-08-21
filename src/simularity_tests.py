
import math
import numpy as np
import seaborn as sns
import pandas
import matplotlib.pyplot as plt
from autoencoders.cnn_with_dense_lts import CNN_DenseLatentSpace
import os
from old_code.file_extractor import FileExtractor

def euclidean_distance(vectorA, vectorB):
    if len(vectorA) != len(vectorB):
        raise ValueError('ERROR MEMORY SYSTEM EUCLIDEAN DISTANCE: '
                         'Input vectors must be of equal length')

    sum = 0
    for pointA, pointB in zip(vectorA, vectorB):
        diff = pointA - pointB
        sum += math.pow(diff, 2)

    result = math.sqrt(sum)
    return result

def normalise(min, max, x):
    return (x-min) / (max - min)



if __name__ == '__main__':

    # OBJECTIVE : Perform Pair wise comparison of latent reprentations based on euclidean simularity
    save_figs = 'test2'
    folderpath = 'autoencoders/models/CNND_main'

    model = CNN_DenseLatentSpace(img_shape=(40, 40, 1),
                                      latent_dimensions=3,
                                      batch_size=1)



    model.load_weights(full_path=folderpath)
    model.data_prep_simple(directory_path='AE_data/sim_test_main_capture', skip_files=[])

    latent_reps = []
    for sample in model.x_train:
        sample = np.reshape(sample, (1,) + sample.shape)
        lr = model.encoder.predict(sample)
        latent_reps.append(lr[0])


    high_simularity_threshold = 0.05
    low_simularity_threshold =  0.7

    comparisons = np.zeros(shape=(len(latent_reps), len(latent_reps)))
    threshold_comparisons = np.zeros(shape=(len(latent_reps), len(latent_reps)))
    for i in range(len(latent_reps)):
        item1 = latent_reps[i]
        for j in range(len(latent_reps)):
            item2 = latent_reps[j]

            simularity = euclidean_distance(item1, item2)
            comparisons[i][j] = simularity


    min = comparisons.min()
    max = comparisons.max()

    func = np.vectorize(normalise)
    comparisons = func(min, max, comparisons)

    for i in range(len(comparisons)):
        for j in range(len(comparisons)):
            sample = comparisons[i][j]

            if sample <= high_simularity_threshold:
                threshold_comparisons[i][j] = 1

            elif sample > high_simularity_threshold and sample <= low_simularity_threshold:
                threshold_comparisons[i][j] = 0

            elif sample > low_simularity_threshold:
                threshold_comparisons[i][j] = -1

    fig1 = plt.figure(1)

    for set in comparisons:
        plt.plot(set, linewidth=2)


    plt.legend(['level_0', 'level_1', 'level_2',
                  'level_3', 'level_4', 'level_5', 'level_6', 'level_7', 'level_8'])

    plt.xlabel('Simulation Environment Number')
    plt.ylabel('Euclidean Distance')
    plt.grid()

    fig2 = plt.figure(2, figsize=(20, 4))

    n = len(model.x_train)
    for i in range(n):
        # disp original

        ax = plt.subplot(2, n, i + 1)
        plt.imshow(model.x_train[i].reshape(40, 40))
        plt.title(str(i))
        plt.gray()

    fig3 = plt.figure(3)
    sns.heatmap(comparisons, annot=True, linewidths=.5)

    fig4 = plt.figure(4)
    sns.heatmap(threshold_comparisons, annot=True, linewidths=.5)





    plt.show()

    if os.path.isdir('figures')==False:
        os.mkdir('figures')

    fig1.savefig(os.path.join('figures', save_figs +'1'))
    fig2.savefig(os.path.join('figures', save_figs+'2'))
    fig3.savefig(os.path.join('figures', save_figs+'3'))
    fig4.savefig(os.path.join('figures', save_figs+'4'))
    print()
    print('MIN VALUE = %s' % min)
    print('MAX VALUE = %s' % max)