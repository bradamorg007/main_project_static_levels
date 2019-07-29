import numpy as np
import pygame
import os
import bz2
import pickle as pkl
import matplotlib.pyplot as plt
import sys
import json


class ImageCapture:

    def __init__(self, buffer_size, max_frames, step_size=1, capture_first_epoch_only=True,
                 capture_mode='save', save_folder_path=None, feed_object=None,
                 grey_scale=True, rescale_shape=None, normalise=True, preview_images=False, show_progress=False):

        self.buffer_size = buffer_size
        self.max_frames = max_frames
        self.step_size = step_size
        self.capture_first_epoch_only = capture_first_epoch_only
        self.capture_mode = capture_mode
        self.save_folder_path = save_folder_path
        self.feed_object = feed_object
        self.grey_scale = grey_scale
        self.rescale_shape = rescale_shape
        self.normalise = normalise
        self.preview_images = preview_images
        self.show_progress = show_progress
        self.batch_sizes = []
        self.batch_number = 0
        self.LOCK_FUNCTIONALITY = False
        self.end_batch_number = 0
        self.folder_bin = 'AE_data'

        self.buffer = []

        if self.capture_mode != 'save' and self.capture_mode != 'feed':
            raise ValueError('ERROR ImageCapture __init__: capture mode must be set to either save or feed')

        if self.capture_mode == 'save' and self.save_folder_path == None:
            raise  ValueError('ERROR ImageCapture __init__: if capture mode is set to save then a valid savepath must stated')

        if self.capture_mode == 'feed' and self.feed_object == None:
            raise ValueError('ERROR ImageCapture __init__: if capture mode is set to feed then a valid feed_object must be stated')

        if self.buffer_size < 0.0 or self.buffer_size > 1.0:
            raise ValueError('ERROR ImageCapture __init__: buffer size must have a value between 0.0 and 1.0')

        if not os.path.isdir(self.folder_bin):
            os.mkdir(self.folder_bin)

        if not os.path.isdir(os.path.join(self.folder_bin, self.save_folder_path)):
            os.mkdir(os.path.join(self.folder_bin, self.save_folder_path))


        self.compute_batch_sizes()


    def compute_batch_sizes(self):

        remainder = 1.0 % self.buffer_size
        full_sizes = int(np.round((1.0 - remainder) / self.buffer_size))

        for i in range(full_sizes):
            self.batch_sizes.append(int(np.round(self.buffer_size * self.max_frames)))

        self.batch_sizes.append(int(np.round(remainder * self.max_frames)))

        self.end_batch_number = len(self.batch_sizes)

        if self.show_progress:
            self.progressBar(0, self.end_batch_number, 20)


    def process(self, surface):

        if self.rescale_shape != None:
            surface = pygame.transform.scale(surface, self.rescale_shape)

        if self.grey_scale:
            surface = self.grayConversion(pygame.surfarray.array3d(surface))

        if self.normalise:
            surface = surface / 255

        surface = surface.swapaxes(1, 0)

        if self.preview_images:
            plt.figure()
            plt.title('Preview of Capture Images')
            plt.imshow(surface)
            plt.gray()
            plt.show()


        return surface


    def save_buffer(self, input, label):

        if len(self.batch_sizes) == 0:
            self.LOCK_FUNCTIONALITY = True
            print('\n ImageCapture : Capture Mode Terminated')
            return

        self.buffer.append([input, label])

        self.batch_sizes[0] -= 1

        if self.batch_sizes[0] == 0:
            self.batch_sizes.pop(0)
            self.save()


    def save(self):

        filename = 'batch_' + str(self.batch_number)
        fullpath = bz2.BZ2File(os.path.join(self.folder_bin, self.save_folder_path, filename), 'w')
        pkl.dump(self.buffer, fullpath)
        fullpath.close()
        self.batch_number += 1
        self.buffer = []

        if self.show_progress:
            self.progressBar(self.batch_number, self.end_batch_number, 20)


    def progressBar(self, value, endvalue, bar_length=20):
        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length) - 1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        sys.stdout.write("\rImage Capture: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush()


    def process_capture_request(self, surface, label):

        surface = self.process(surface)

        # save or feed image out
        if self.capture_mode == 'save':
            self.save_buffer(input=surface, label=label)


    def capture(self, surface, label, level_manager, frame_count):


        if level_manager.mode == 'capture' and level_manager.capture_mode_override == False:

            if self.LOCK_FUNCTIONALITY == False:

                if frame_count % self.step_size == 0:

                    if self.capture_first_epoch_only:

                        # preprocess image
                        if level_manager.epoch_count <= 1:
                            self.process_capture_request(surface, label)


                    else:
                        self.process_capture_request(surface, label)


    def grayConversion(self, image):
        grayValue = 0.07 * image[:,:,2] + 0.72 * image[:,:,1] + 0.21 * image[:,:,0]
        gray_img = grayValue.astype(np.uint8)
        return gray_img


    def save_config(self, save_folder_path):

        if not os.path.isdir(os.path.join(self.folder_bin, save_folder_path)):
            os.mkdir(os.path.join(self.folder_bin, save_folder_path))

        dict = self.__dict__
        filename = os.path.join(self.folder_bin, save_folder_path, 'image_capture_config.json')

        with open(filename, 'w') as f:
            json.dump(dict, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    pass