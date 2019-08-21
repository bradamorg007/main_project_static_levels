import pygame
import os
import math
import pickle as pkl
from level import Level
from block import Block
from agent_manager import AgentManager
from genetic_algorithm import GenticAlgorithm
import matplotlib.pyplot as plt
import numpy as np
import bz2
from visual_system import VisualSystem
from memory_system import MemorySystem

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
PURPLE = (255, 0, 255)
SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 600

#'[400, 460, 560, 100, BLACK]
def build_blueprint_range(x, start_bottom_y, start_top_y, width, stop=0, multipliers=1, color=BLACK, y_interval=1, x_interval=0):


    if start_bottom_y > SCREEN_HEIGHT or start_top_y > SCREEN_HEIGHT or start_bottom_y < 0 or start_top_y < 0:
        raise ValueError('ERROR: Invalid start_bottom_y or start_top_y')

    if y_interval == 0:
        y_interval = 1
    elif y_interval > 0:
        y_interval = y_interval * -1

    output_array = []
    start_top_y_list = np.arange(start=start_top_y, stop=stop, step=y_interval)
    stop = stop + start_bottom_y - start_top_y
    start_bottom_y_list = np.arange(start=start_bottom_y, stop=stop, step=y_interval)

    if len(start_bottom_y_list) != len(start_top_y_list):
        raise ValueError('ERROR: start_top_y_list and start_bottom_y_list are not th same lengths')

    for i in range(len(start_bottom_y_list)):

        blueprint = [x, int(start_top_y_list[i]), int(start_bottom_y_list[i]), width, color]
        blueprints = build_blueprint(blueprint, multipliers=multipliers, xPos_interval=x_interval)
        output_array.append(blueprints)
        # for item in blueprints:
        #     output_array.append(item)



    return output_array



def build_blueprint(blueprint, multipliers, xPos_interval=0):

        level_array = []

        new_Xpos = 0
        pos = blueprint[0]
        for i in range(multipliers):


            if i > 0:
                new_blueprint = blueprint.copy()
                new_blueprint[0] = pos
                new_Xpos += blueprint[3] + xPos_interval
                new_blueprint[0] += new_Xpos

                if blueprint[0] < SCREEN_WIDTH:
                    level_array.append(new_blueprint)

            else:
                level_array.append(blueprint)

        level_array[len(level_array)-1][4] = GREEN

        return level_array


def merge(*args):

    output = []

    for item in args:

        output = output + item

    return output


def init(screen_width, screen_height, levels):

    # Call this function so the Pygame library can initialize itself
    pygame.init()

    # Create an 800x600 sized screen
    screen = pygame.display.set_mode([screen_width, screen_height])

    # Set the title of the window
    pygame.display.set_caption('ML Simulation')


    current_level_no = 0
    current_level = levels[current_level_no]

    clock = pygame.time.Clock()

    return screen, clock, current_level, current_level_no


def save_agent(agent):
    # save best agent
    agent.computeFitness()
    folder = "sim_data"
    filename = os.path.join(folder, 'best_agent')
    if os.path.isdir(folder) == False:
        os.mkdir(folder)

    pkl.dump(agent.functional_system, open(os.path.join(filename), 'wb'))
    print('NOTIFICATION: Best Agent has been saved to %s' % filename)


def grayConversion(image):
    grayValue = 0.07 * image[:,:,2] + 0.72 * image[:,:,1] + 0.21 * image[:,:,0]
    gray_img = grayValue.astype(np.uint8)
    return gray_img


def capture(surface, mode, level_number, rescale_shape=(40,40), grayscale=True,
            normalise=True, preview=False, remove_blank_frames=True, save_folder_path='default_save'):

    capture_success = False
    if mode == 'capture':
        surface = pygame.transform.scale(surface, rescale_shape)

        if grayscale:
            surface = grayConversion(pygame.surfarray.array3d(surface))

        skip = False
        if remove_blank_frames:
            if surface.max() == surface.min():
                skip = True

        if skip == False:
            if surface.max() != surface.min():
                if normalise:
                    surface = surface / 255

                surface = surface.swapaxes(1, 0)

                if preview:
                    plt.figure()
                    plt.title('Preview of Capture Images')
                    plt.imshow(surface)
                    plt.gray()
                    plt.show()


                folder_bin = 'AE_validation_data'
                if not os.path.isdir(folder_bin):
                    os.mkdir(folder_bin)

                if not os.path.isdir(os.path.join(folder_bin, save_folder_path)):
                    os.mkdir(os.path.join(folder_bin, save_folder_path))

                filename = 'level_' + str(level_number)
                fullpath = bz2.BZ2File(os.path.join(folder_bin, save_folder_path, filename), 'w')
                pkl.dump(surface, fullpath)
                fullpath.close()
                capture_success = True
    return capture_success


def run_model(x, start_bottom_y,start_top_y, step, multipliers, x_interval,  save_name):

    """ Main Program """
    stop = start_top_y
    count = 1
    while stop > 0:

        stop -= step
        print_data = True

        save_folder_path = save_name + str(count)


        test_levels = build_blueprint_range(x=x, start_bottom_y=start_bottom_y, start_top_y=start_top_y, width=100,
                                            stop=stop, multipliers=multipliers, y_interval=1, x_interval=x_interval)


        #level1
        levels = Block.generate(SCREEN_WIDTH, SCREEN_HEIGHT, 100, True, test_levels)

        screen, clock, current_level, current_level_no = init(SCREEN_WIDTH, SCREEN_HEIGHT, levels)


        done = False
        while not done:

            # --- Event Processing ---

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True


            # --- Drawing ---
            screen.fill(WHITE)

            for block in current_level:
                block.draw(screen)


            capture(surface=screen, mode='capture', remove_blank_frames=True,
                    level_number=current_level_no,
                    save_folder_path=save_folder_path, preview=False)

            if current_level_no < len(levels) - 1:
                current_level_no += 1
                current_level = levels[current_level_no]
            else:
                done = True


            pygame.display.flip()

            clock.tick(60)



        if print_data:
            print('Simulation Terminated')
        pygame.quit()
        count += 1

if __name__ == "__main__":

    # 600, 500
    #-125
    run_model()