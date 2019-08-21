import os
from autoencoders.variational_autoencoder_symmetric import VaritationalAutoEncoderSymmetric
from autoencoders.cnn_with_dense_lts import CNN_DenseLatentSpace
import pygame
import matplotlib.pyplot as plt
import numpy as np


class VisualSystem:

    def __init__(self, img_shape, latent_dims, RE_delta, model_folder, start=0, preview_output=False, preview_topology=False):

        self.model_dir  = 'autoencoders/models/'
        self.model_folder = model_folder
        self.RE_delta = RE_delta
        self.start = start
        self.framecount = 0
        self.preview_output = preview_output
        self.supress = False

        self.full_functionaility_enabled = False
        self.model = CNN_DenseLatentSpace(img_shape=img_shape,
                                                      latent_dimensions=latent_dims,
                                                      batch_size=1, print_summary=preview_topology)

        self.model.load_weights(full_path=os.path.join(self.model_dir, self.model_folder))

        self.current_input_sample = None


    def clean(self):
        self.current_input_sample = None


    def process_image(self, surface, agents, frame_count, mask_agent=True, preview_images=False):

        if self.start <= frame_count:

            if mask_agent:
                # remove agent from image
                xPos, yPos = [agents.not_sprites[0].rect.x, agents.not_sprites[0].rect.y]
                radius = agents.not_sprites[0].radius
                w_loc = np.arange(xPos, xPos + radius + 1)
                y_loc = np.arange(yPos, yPos + radius + 1)

                pix_array =  pygame.surfarray.pixels3d(surface)
                for x in w_loc:
                    for y in y_loc:
                        pix_array[x][y][:] = 255

            surface = pygame.transform.scale(surface, (self.model.img_shape[0],self.model.img_shape[1]))

            surface = self.grayConversion(pygame.surfarray.array3d(surface))

            surface = surface / 255

            surface = surface.swapaxes(1, 0)

            if preview_images:
                plt.figure()
                plt.title('Preview of Capture Images')
                plt.imshow(surface)
                plt.gray()
                plt.show()

            surface = surface.reshape((1,) + surface.shape + (1,))

            self.current_input_sample = surface
            self.full_functionaility_enabled = True



    def grayConversion(self, image):
        grayValue = 0.07 * image[:,:,2] + 0.72 * image[:,:,1] + 0.21 * image[:,:,0]
        gray_img = grayValue.astype(np.uint8)
        return gray_img


    def is_familiar(self):

        if self.full_functionaility_enabled:
            self.full_functionaility_enabled = False

            RE = self.model.model.evaluate(self.current_input_sample, self.current_input_sample, verbose=0)

            if RE > self.model.reconstruction_error + self.RE_delta:
                pred = self.model.model.predict(self.current_input_sample)

                if self.preview_output:
                    plt.figure()
                    plt.imshow(pred.reshape(40, 40))
                    plt.title('prediction')
                    plt.gray()

                    plt.figure()
                    plt.imshow(self.current_input_sample.reshape(40, 40))
                    plt.title('original')
                    plt.gray()
                    plt.show()

                return False
            else:
                return True



    def generate_latent_representation(self):

        ls = self.model.encoder.predict(self.current_input_sample)

        output1 = ls[0]
        output2 = None

        if len(ls) == 2:
            if len(output1.shape) == 2:
                output1 = output1[0]
            output2 = ls[1]

        return output1, output2

    @staticmethod
    def init(img_shape, latent_dims, RE_delta,
                     model_folder, start, MODE,preview_output,preview_topology):

        if MODE == 'test':
           vs =  VisualSystem(img_shape, latent_dims, RE_delta,
                         model_folder, start, preview_output=preview_output, preview_topology=preview_topology)

           return vs

        else:
            return None

if __name__ == '__main__':
    pass