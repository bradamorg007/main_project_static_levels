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

def normalise(min, max, x):
    return (x-min) / (max - min)

def run_simulation(DRAW, SCREEN_HEIGHT, SCREEN_WIDTH,
                   MODE, EXTRACT_DATA, SIM_SPEED,
                   save_first_winner, level_to_save,
                   generation_limit, end_on_lap,
                   capture_filename, levels, attempts,
                   save_trained_agent, load_agent_filepath,
                   high_simularity_threshold,
                   low_simularity_threshold,
                   min_lts, max_lts,
                   visual_system):


    print()
    print('MODEL = *** STM SEARCH ***')
    print()

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
                           rand_init_population_size=300,
                           rand_init_mutation_rate=0.0,
                           med_sim_init_population_size=300,
                           med_sim_init_rand_agent_percenatge=0.1,
                           med_sim_init_med_memory_agent_percenatge=0.9,
                           med_sim_init_rand_mutation_rate=0.0,
                           med_sim_init_med_mutation_rate=0.3,
                           load_agent_filepath=load_agent_filepath,
                           save_path=save_trained_agent
                         )

    # visual_system = VisualSystem.init(img_shape=(40, 40, 1),
    #                                   latent_dims=3,
    #                                   RE_delta=0.0,
    #                                   model_folder='CNND_main',
    #                                   start=1,
    #                                   MODE=MODE,
    #                                   preview_output=False
    #                                  )

    memory_system = MemorySystem.init(MODE=MODE,
                                      high_simularity_threshold=high_simularity_threshold,
                                      low_simularity_threshold =low_simularity_threshold,
                                      forget_usage_threshold=0,
                                      forget_age_threshold=50,
                                      max_memory_size=50
                                     )

    screen, clock, current_level, current_level_no = init(SCREEN_WIDTH, SCREEN_HEIGHT, levels)
    norm = np.vectorize(normalise)

    internal_generation_count = 0
    level_generation_count = 0
    attempts = attempts
    starting_attempts = attempts
    generation_record = np.zeros(len(levels)) - 1
    generation_record_count = 0
    lap_no = 1
    change_level_flag = False
    winners = []
    norm_latent_rep = []
    is_familiar = None
    init_first_original_memory = True
    memory = None

    event_flag = False
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

                elif event.key == pygame.K_d:
                    if DRAW:
                        DRAW = False
                    else:
                        DRAW = True

                elif event.key == pygame.K_SPACE:
                    pause = True
                    while pause == True:
                        for event in pygame.event.get():
                            if event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_SPACE:
                                    pause = False




        for j in range(SIM_SPEED):

            if MODE == 'train' or 'test':
                # --- Update Agents ---
                i = 0
                while i < len(agents.not_sprites) and i > -1:
                    agent = agents.not_sprites[i]

                    if MODE == 'test' and visual_system is not None:
                        # ignore blackout screen at beginning
                        visual_system.process_image(screen, agents, frame_count, mask_agent=True, preview_images=False)
                        is_familiar = visual_system.is_familiar()

                        if is_familiar and is_familiar is not None and init_first_original_memory == True:
                            latent_representation, _ = visual_system.generate_latent_representation()
                            norm_latent_rep = norm(min_lts, max_lts, latent_representation)
                            memory_system.create_memory(latent_representation=norm_latent_rep,
                                                        solution=agent.functional_system,
                                                        tag='mem_level' + str(current_level_no))
                            init_first_original_memory = False
                            visual_system.supress = True
                            event_flag = True

                        if visual_system.supress==False and is_familiar is not None:
                            # get latent representation from vs
                            latent_representation, _ = visual_system.generate_latent_representation()
                            norm_latent_rep = norm(min_lts, max_lts, latent_representation)
                            memory, action = memory_system.query(norm_latent_rep)

                            if memory is not None and action == 'memory_to_fs_system_switch':
                                agents.temp_agent_store.append(agent)
                                agent = agents.reset_temp_agent_store(memory.get('solution'))
                                visual_system.supress = True
                                print(action)

                            elif memory is not None and action == 'adaption_using_medium_memory_as_init_foundation':
                                MODE = 'adapt'
                                agents.adaptive_med_sim_population_init(MODE, memory)
                                print(action)


                            elif action == 'adaption_using_low_memory_and_random_init_foundation':
                                MODE = 'adapt'
                                agents.adaptive_rand_population_init(MODE) # leave one model the current one unchanges
                                print(action)

                            event_flag = True



                     #   print('visual system is frame familiar: %s' % is_familiar)


                    agent.think(current_level, SCREEN_WIDTH, SCREEN_HEIGHT)
                    agent.update(SCREEN_HEIGHT)

                    if agent.current_closest_block.hit(agent) or agent.off_screen(SCREEN_HEIGHT, SCREEN_WIDTH) or agent.fuel_depleted():

                       # agent.computeFitness()
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

                        last_level = False
                        if current_level_no == len(levels)-1:
                            last_level = True

                        if current_level_no < len(levels)-1:
                            current_level_no += 1
                        else:
                            current_level_no = 0

                        current_level = levels[current_level_no]


                        level_generation_count += internal_generation_count
                        generation_record[generation_record_count] = level_generation_count
                        generation_record_count += 1
                        level_generation_count = 0

                        if last_level:
                            if current_level_no == 0 and end_on_lap == lap_no:
                                done = True
                                break
                            else:
                                lap_no += 1

                        if visual_system is not None and visual_system.supress == True:
                            visual_system.supress = False


                    else:
                        solution = get_best_winner(winners)
                        memory_system.create_memory(latent_representation=norm_latent_rep,
                                                    solution=solution.functional_system, tag='mem_level'+str(current_level_no))
                        agents.reset_temp_agent_store(solution.functional_system)
                        agents.not_sprites[0].reset()
                        visual_system.supress = True
                        level_generation_count += internal_generation_count
                        internal_generation_count = 0
                        winners = []
                        agents.dead_agents = []
                        print('LEN' + str(len(agents.not_sprites)))
                        MODE = 'test'
                        print('LEVEL ' + str(current_level_no) + 'ADAPT COMPLETE STM: SOLUTION: %s' % (solution.functional_system.name))




                # check if all active agents are dead, the perform GA and reset game level and epochs
                if len(agents.not_sprites) == 0:

                    skip_ga = False
                    if MODE == 'test':
                        done = True
                        break

                        # dead_agent = agents.dead_agents[0]
                        # dead_agent.reset()
                        # new_population = [dead_agent]
                        # agents.update_arrays(input=new_population)
                        # MODE = 'adapt'
                        # agents.dead_agents = []
                        # agents.adaptive_med_sim_population_init(MODE, memory)
                        # skip_ga = True
                       # raise ValueError('AGENT DEATH DURING TEST MODE: MEMORY WAS INEFFECTIVE')

                    else:

                        if MODE == 'adapt' and internal_generation_count >= generation_limit:

                            if memory_system.current_action == 'adaption_using_medium_memory_as_init_foundation':
                                MODE = 'adapt'
                                agents.dead_agents = []
                                agents.adaptive_med_sim_population_init(MODE, memory)

                            elif memory_system.current_action == 'adaption_using_low_memory_and_random_init_foundation':
                                MODE = 'adapt'
                                agents.dead_agents = []
                                agents.adaptive_rand_population_init(MODE)  # leave one model the current one unchanges
                            skip_ga = True

                            level_generation_count += internal_generation_count
                            internal_generation_count = 0

                            if EXTRACT_DATA == 'gen_test' and attempts == 0:
                                generation_record[generation_record_count] = level_generation_count
                                done = True
                                print('~FAIL')
                                break

                            else:
                                attempts -= 1

                            print('MODEL LOST IN BAD THOUGHT POOL: ATTEMPTS = %s out %s RE-INIT GA PROCESSING' % (
                            attempts, starting_attempts))

                            print('MODEL LOST IN BAD THOUGHT POOL: RE-INIT GA PROCESSING')

                    if skip_ga == False:

                        new_population = GenticAlgorithm.produceNextGeneration(population=agents.dead_agents,
                                                                      agent_meta_data=agents.__dict__)

                        agents.update_arrays(new_population)

                        internal_generation_count += 1
                        print('generation = %s population size = %s level no = %s / %s' % (
                            internal_generation_count, len(new_population), current_level_no, len(levels)))

                        # if MODE != 'adapt':
                        #     current_level_no = 0
                        #     current_level = levels[current_level_no]







            # --- Drawing ---

            screen.fill(WHITE)

            if DRAW:
                for block in current_level:
                    block.draw(screen)

                agents.draw(screen, mode=MODE)


            capture_success = capture(surface=screen, mode=MODE, remove_blank_frames=True,
                    level_number=current_level_no,
                    save_folder_path=capture_filename, preview=True)

            if MODE == 'capture' and capture_success:
                if current_level_no < len(levels) - 1:
                    current_level_no += 1
                    current_level = levels[current_level_no]
                else:
                    done = True

            pygame.display.flip()
            clock.tick()
            frame_count += 1


            if change_level_flag:
                change_level_flag = False

            if event_flag:

                if memory is None:
                    sim_score = 'N/A'
                else:

                    if isinstance(memory, list):
                        min = memory[0].get('similarity_score')

                        for item in memory:
                            sim = item.get('similarity_score')
                            if sim < min:
                                min = sim
                        sim_score = 'minimum = ' + str(min)

                    else:
                        sim_score =  memory.get('similarity_score')

                if MODE == 'adapt':
                    print('Mode: %s is familiar: %s model in use: %s similarity: %s' % (MODE, is_familiar, 'Finding solution', sim_score))

                elif MODE == 'test':
                    print('Mode: %s is familiar: %s model in use: %s Similarity: %s' % (MODE, is_familiar, agents.not_sprites[0].functional_system.name,
                                                                                       sim_score))

                event_flag = False

    pygame.quit()
    print('SIMULATION LEVEL COMPLETED BY AGENT')
    return generation_record

if __name__ == "__run_simulation__":
    #gen_rec = run_simulation()
    pass