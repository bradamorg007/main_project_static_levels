import pygame
import os
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


def build_blueprint(blueprint, multipliers, xPos_interval=None):

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


                folder_bin = 'AE_data'
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

def get_best_winner(winners):
    winners[0].computeFitness()
    max = winners[0].fitness
    max_winner = winners[0]

    for winner in winners:
        winner.computeFitness()
        fitness = winner.fitness
        if fitness > max:
            max = fitness
            max_winner = winner

    return max_winner



def main():
    """ Main Program """
    MODE = 'test'
    SIM_SPEED = 1
    save_first_winner = False
    level_to_save = 0


    level1 = build_blueprint([400, 450, 580, 100, BLACK], multipliers=4, xPos_interval=50)
    level2 = build_blueprint([400, 400, 520, 100, BLACK], multipliers=4, xPos_interval=50)
    level3 = build_blueprint([400, 350, 480, 100, BLACK], multipliers=4, xPos_interval=50)
    level4 = build_blueprint([400, 250, 380, 100, BLACK], multipliers=4, xPos_interval=50)
    level5 = build_blueprint([400, 150, 280, 100, BLACK], multipliers=4, xPos_interval=50)
    level6 = build_blueprint([400, 50,  180, 100, BLACK], multipliers=4, xPos_interval=50)

    levels = Block.generate(SCREEN_WIDTH, SCREEN_HEIGHT, 100, level1, level3, level1, level4, level6)
    #levels = Block.generate(SCREEN_WIDTH, SCREEN_HEIGHT, 100, level1)

    agents = AgentManager( mode=MODE,
                           yPos_range=None,
                           starting_xPos=100,
                           starting_yPos=500,
                           xPos_range=None,
                           population_size=300,
                           screen_height=SCREEN_HEIGHT,
                           vertical_fuel_depletion_rate=0.0005,
                           horizontal_fuel_depletion_rate=0.0005,
                           screen_width=SCREEN_WIDTH,
                           load_agent_filepath='sim_data/default_best'
                         )

    visual_system = VisualSystem.init(img_shape=(40, 40, 1),
                                      latent_dims=3,
                                      RE_delta=0.0,
                                      model_folder='CNND',
                                      start=6,
                                      MODE=MODE,
                                      preview_output=False
                                     )

    memory_system = MemorySystem.init(MODE=MODE,
                                      high_simularity_threshold=1.0e-08,
                                      low_simularity_threshold=0.2,
                                      forget_usage_threshold=0,
                                      forget_age_threshold=50,
                                      max_memory_size=50
                                     )

    screen, clock, current_level, current_level_no = init(SCREEN_WIDTH, SCREEN_HEIGHT, levels)

    generation_count = 0
    change_level_flag = False
    adaption_complete_flag = False
    winners = []
    latent_representation = []
    is_familiar = None

    done = False
    frame_count = 0
    while not done:

        # --- Event Processing ---

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    SIM_SPEED -= 10

                elif event.key == pygame.K_UP:
                    SIM_SPEED += 10

                elif event.key == pygame.K_s and MODE == 'train':
                   agents.save_best_agent()


        for i in range(SIM_SPEED):

            if MODE == 'train' or 'test':
                # --- Update Agents ---
                i = 0
                while i < len(agents.not_sprites) and i > -1:
                    agent = agents.not_sprites[i]

                    if MODE == 'test' and visual_system is not None:
                        # ignore blackout screen at beginning
                        visual_system.process_image(screen, agents, frame_count, preview_images=False)
                        is_familiar = visual_system.is_familiar()

                        if is_familiar == False and memory_system is not None:
                            if visual_system.supress==False:
                                # get latent representation from vs
                                latent_representation, _ = visual_system.generate_latent_representation()
                                memory, action = memory_system.query(latent_representation)

                                if memory is not None and action == 'memory_to_fs_system_switch':
                                    agents.temp_agent_store.append(agent)
                                    agents.reset_temp_agent_store(memory.get('solution'))
                                    visual_system.supress = True
                                    print(action)
                                elif memory is None and action == 'adaption_using_medium_memory_as_init_foundation':
                                    pass

                                elif memory is None and action == 'adaption_using_low_memory_and_random_init_foundation':
                                    MODE = 'adapt'
                                    agents.adaptive_rand_population_init(mode=MODE, population_size=400,
                                                                         rand_to_current_model_split=(1.0, 0.0),
                                                                         rand_model_mutation_rate=0.0,
                                                                         current_model_mutation_rate=0.99) # leave one model the current one unchanges
                                    print('WHERRRRR')
                        else:

                            agents.restore_original_test_agent()



                     #   print('visual system is frame familiar: %s' % is_familiar)



                    agent.think(current_level, SCREEN_WIDTH, SCREEN_HEIGHT)
                    agent.update(SCREEN_HEIGHT)

                    if agent.current_closest_block.hit(agent) or agent.off_screen(SCREEN_HEIGHT, SCREEN_WIDTH) or agent.fuel_depleted():

                        agent.computeFitness()
                        agents.splice(i)
                        i -= 1


                    if len(winners) == 0:
                        if agent.rect.right > current_level[len(current_level) - 1].top_block.rect.right - 10:
                            if save_first_winner == True and level_to_save == current_level_no and MODE=='train':
                                agents.save_best_agent()
                                save_first_winner = False
                            change_level_flag = True
                            agent.fuel = 1.0
                            agent.rect.x = agents.starting_xPos
                            agent.rect.y = agents.starting_yPos
                            agent.velocity_x = 0
                            agent.velocity_y = 0
                            agent.name = 'winner' + str(i)
                            winners.append(agent)
                            agents.splice(i, mode='delete')
                            i -= 1

                    else:
                        if agent.rect.right > current_level[len(current_level)-1].top_block.rect.left and agent.rect.right < current_level[len(current_level)-1].top_block.rect.right:

                            change_level_flag = True
                            agent.fuel = 1.0
                            agent.rect.x = agents.starting_xPos
                            agent.rect.y = agents.starting_yPos
                            agent.velocity_x = 0
                            agent.velocity_y = 0
                            agent.name = 'winner' + str(i)
                            winners.append(agent)
                            agents.splice(i, mode='delete')
                            i -= 1


                    i += 1


                if change_level_flag:

                    if MODE != 'adapt':
                        agents.splice(index=None, mode='all')
                        agents.not_sprites = winners
                        change_level_flag = False
                        winners = []

                        if current_level_no < len(levels)-1:
                            current_level_no += 1
                        else:
                            current_level_no = 0

                        current_level = levels[current_level_no]

                        if visual_system is not None and visual_system.supress == True:
                            visual_system.supress = False
                        break

                    else:
                        solution = get_best_winner(winners)
                        memory_system.create_memory(latent_representation=latent_representation,
                                                    solution=solution.functional_system, tag='mem_level'+str(current_level_no))
                        agents.reset_temp_agent_store(solution.functional_system)
                        visual_system.supress = True
                        winners = []
                        agents.dead_agents = []
                        MODE = 'test'
                        print('ADAPTION COMPLETE: USING NEW SOLUTION')





                # check if all active agents are dead, the perform GA and reset game level and epochs
                if len(agents.not_sprites) == 0:

                    if MODE == 'test':
                        dead_agent = agents.dead_agents[0]
                        dead_agent.reset()
                        new_population = [dead_agent]
                        agents.update_arrays(input=new_population)


                    else:
                        if MODE == 'adapt':
                            a = 0
                        new_population = GenticAlgorithm.produceNextGeneration(population=agents.dead_agents,
                                                                  screen_width=SCREEN_WIDTH,
                                                                  screen_height=SCREEN_HEIGHT,
                                                                  agent_meta_data=agents.__dict__)

                        agents.update_arrays(new_population)

                    generation_count += 1
                    print('generation = %s population size = %s level no = %s / %s' % (
                        generation_count, len(new_population), current_level_no, len(levels)))

                    if MODE != 'adapt':
                        current_level_no = 0
                        current_level = levels[current_level_no]


        # --- Drawing ---
        screen.fill(WHITE)

        for block in current_level:
            block.draw(screen)

        agents.draw(screen, mode=MODE)


        capture_success = capture(surface=screen, mode=MODE, remove_blank_frames=True,
                level_number=current_level_no,
                save_folder_path='test', preview=True)

        if MODE == 'capture' and capture_success:
            if current_level_no < len(levels) - 1:
                current_level_no += 1
                current_level = levels[current_level_no]
            else:
                done = True

        pygame.display.flip()

        clock.tick(60)
        frame_count += 1
        if change_level_flag:
            change_level_flag = False

        if MODE == 'adapt':
            print('Mode: %s is familiar: %s model in use: %s ' % (MODE, is_familiar, 'Finding solution'))

        elif MODE == 'test':
            print('Mode: %s is familiar: %s model in use: %s ' % (MODE, is_familiar, agents.not_sprites[0].functional_system.name))

    pygame.quit()
    print('SIMULATION LEVEL COMPLETED BY AGENT')

if __name__ == "__main__":
    main()