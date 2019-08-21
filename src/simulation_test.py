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



def run_model(MODE=None, data_extract=None, SIM_SPEED=None, generation_limit=None, run_time=None, target_levels=None,
              print_data=True,gentic_algorithm_active=True, target_generation=1,
              med_sim_init_rand_agent_percenatge=1.0, visual_system = None,
              med_sim_init_med_memory_agent_percenatge=0.0,
              med_sim_init_rand_mutation_rate=0.0,
              med_sim_init_med_mutation_rate=0.5
              ):
    """ Main Program """
    if data_extract is None:
        MODE = 'capture'
        gentic_algorithm_active = True
        data_extract = False
        SIM_SPEED = 1
        generation_limit = 10
        target_generation = 3
        run_time = 600
        target_levels = [1]
        print_data = True

        med_sim_init_rand_agent_percenatge = 0.0
        med_sim_init_med_memory_agent_percenatge = 1.0
        med_sim_init_rand_mutation_rate = 0.0
        med_sim_init_med_mutation_rate = 0.1

    save_first_winner = False
    level_to_save = 0
    save_folder_path = 'test_report'

    test_levels = build_blueprint_range(x=400, start_bottom_y=600, start_top_y=500, width=100,
                                        stop=0, multipliers=4, y_interval=1, x_interval=50)
 #390, 490
    level0 = build_blueprint([400, 460, 560, 100, BLACK], multipliers=4, xPos_interval=50)
    level1 = build_blueprint([400, 390, 490, 100, BLACK], multipliers=4, xPos_interval=50)
    level2 = build_blueprint([400, 350, 450, 100, BLACK], multipliers=4, xPos_interval=50)
    level3 = build_blueprint([400, 330, 430, 100, BLACK], multipliers=4, xPos_interval=50)
    level4 = build_blueprint([400, 250, 350, 100, BLACK], multipliers=4, xPos_interval=50)

    level0 = build_blueprint([300, 500, 600, 100, BLACK], multipliers=1, xPos_interval=100)
    level1 = build_blueprint([300, 480, 580, 100, BLACK], multipliers=2, xPos_interval=90)
    level2 = build_blueprint([300, 460, 560, 100, BLACK], multipliers=3, xPos_interval=80)
    level3 = build_blueprint([300, 440, 540, 100, BLACK], multipliers=4, xPos_interval=70)
    level4 = build_blueprint([300, 420, 520, 100, BLACK], multipliers=5, xPos_interval=60)
    level5 = build_blueprint([300, 400, 500, 100, BLACK], multipliers=6, xPos_interval=50)
    level6 = build_blueprint([300, 380, 480, 100, BLACK], multipliers=7, xPos_interval=40)
    level7 = build_blueprint([300, 360, 460, 100, BLACK], multipliers=8, xPos_interval=30)
    level8 = build_blueprint([300, 340, 440, 100, BLACK], multipliers=9, xPos_interval=20)
    level9 = build_blueprint([300, 320, 420, 100, BLACK], multipliers=10, xPos_interval=10)
    level10 =build_blueprint([300, 300, 400, 100, BLACK], multipliers=11, xPos_interval=0)


    #level1
    levels = Block.generate(SCREEN_WIDTH, SCREEN_HEIGHT, 100, False, level0, level1,level2, level3, level4,
                            level5, level6,level7, level8, level9, level10)
    #levels = Block.generate(SCREEN_WIDTH, SCREEN_HEIGHT, 100, level1)

    agents = AgentManager( mode=MODE,
                           yPos_range=None,
                           starting_xPos=100,
                           starting_yPos=500,
                           xPos_range=None,
                           population_size=300,
                           screen_height=SCREEN_HEIGHT,
                           vertical_fuel_depletion_rate=0.001,
                           horizontal_fuel_depletion_rate=0.001,
                           screen_width=SCREEN_WIDTH,
                           rand_init_population_size=300,
                           rand_init_mutation_rate=0.0,
                           med_sim_init_population_size=300,
                           med_sim_init_rand_agent_percenatge=med_sim_init_rand_agent_percenatge,
                           med_sim_init_med_memory_agent_percenatge=med_sim_init_med_memory_agent_percenatge,
                           med_sim_init_rand_mutation_rate=med_sim_init_rand_mutation_rate,
                           med_sim_init_med_mutation_rate=med_sim_init_med_mutation_rate,
                           load_agent_filepath='sim_data/default_best'
                         )



    if visual_system == None:
        visual_system = VisualSystem.init(img_shape=(40, 40, 1),
                                          latent_dims=3,
                                          RE_delta=0.0,
                                          model_folder='CNND_sim_test',
                                          start=1,
                                          MODE=MODE,
                                          preview_output=False
                                         )

    memory_system = MemorySystem.init(MODE=MODE,
                                      high_simularity_threshold=0.05,
                                      low_simularity_threshold = 0.09,
                                      forget_usage_threshold=0,
                                      forget_age_threshold=50,
                                      max_memory_size=50
                                     )

    screen, clock, current_level, current_level_no = init(SCREEN_WIDTH, SCREEN_HEIGHT, levels)

    avg_fitness = None


    generation_count = 0
    global_generation_count = 0
    change_level_flag = False
    winners = []
    latent_representation = []
    is_familiar = None
    init_first_original_memory = True
    memory = None

    event_flag = False
    done = False
    frame_count = 0
    start_time = pygame.time.get_ticks()
    if print_data:
        print('MODEL IS ACTIVE: GA ACTIVE = %s' % gentic_algorithm_active)
    while not done:

        is_target_level = False
        if  data_extract:
            for targets in target_levels:
                if current_level_no == targets:
                    is_target_level = True
                    break

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

                elif event.key == pygame.K_SPACE:
                    pause = True
                    while pause == True:
                        for event in pygame.event.get():
                            if event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_SPACE:
                                    pause = False



        for i in range(SIM_SPEED):

            if MODE == 'train' or MODE == 'test':
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
                            memory_system.create_memory(latent_representation=latent_representation,
                                                        solution=agent.functional_system,
                                                        tag='mem_level' + str(current_level_no))
                            init_first_original_memory = False
                            visual_system.supress = True
                            event_flag = True

                        if visual_system.supress==False and is_familiar is not None:
                            # get latent representation from vs
                            latent_representation, _ = visual_system.generate_latent_representation()
                            memory, action = memory_system.query(latent_representation)

                            if memory is not None and action == 'memory_to_fs_system_switch':
                                agents.temp_agent_store.append(agent)
                                agent = agents.reset_temp_agent_store(memory.get('solution'))
                                visual_system.supress = True
                                if print_data:
                                    print(action)

                            elif memory is not None and action == 'adaption_using_medium_memory_as_init_foundation':
                                MODE = 'adapt'
                                agents.adaptive_med_sim_population_init(MODE, memory)
                                if print_data:
                                    print(action)
                                break


                            elif action == 'adaption_using_low_memory_and_random_init_foundation':
                                MODE = 'adapt'
                                agents.adaptive_rand_population_init(MODE) # leave one model the current one unchanges
                                if print_data:
                                    print(action)
                                break

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

                        if current_level_no < len(levels)-1:
                            current_level_no += 1
                        else:
                            current_level_no = 0

                        current_level = levels[current_level_no]

                        if visual_system is not None and visual_system.supress == True:
                            visual_system.supress = False
                        break

                    else:

                        if is_target_level:
                            sum_fitness = 0

                            for ag in agents.dead_agents:
                                ag.computeFitness()
                                sum_fitness += ag.fitness

                            for ag in winners:
                                ag.computeFitness()
                                sum_fitness += ag.fitness

                            avg_fitness = sum_fitness / (len(agents.dead_agents) + len(winners))
                            done = True
                            break

                        solution = get_best_winner(winners)
                        memory_system.create_memory(latent_representation=latent_representation,
                                                    solution=solution.functional_system, tag='mem_level'+str(current_level_no))
                        agents.reset_temp_agent_store(solution.functional_system)
                        visual_system.supress = True
                        generation_count = 0
                        winners = []
                        agents.dead_agents = []
                        MODE = 'test'

                        if print_data:
                            print('ADAPTION COMPLETE: SOLUTION: %s' % (solution.functional_system.name))



                # check if all active agents are dead, the perform GA and reset game level and epochs
                if len(agents.not_sprites) == 0:

                    skip_ga = False
                    if MODE == 'test':

                        dead_agent = agents.dead_agents[0]
                        dead_agent.reset()
                        new_population = [dead_agent]
                        agents.update_arrays(input=new_population)
                        return 'ERROR_AGENT_DEATH'
                        #raise ValueError('AGENT DEATH DURING TEST MODE: MEMORY WAS INEFFECTIVE')

                    else:

                        if MODE == 'adapt' and generation_count >= generation_limit:

                            if memory_system.current_action == 'adaption_using_medium_memory_as_init_foundation':
                                MODE = 'adapt'
                                agents.dead_agents = []
                                agents.adaptive_med_sim_population_init(MODE, memory)

                            elif memory_system.current_action == 'adaption_using_low_memory_and_random_init_foundation':
                                MODE = 'adapt'
                                agents.dead_agents = []
                                agents.adaptive_rand_population_init(MODE)  # leave one model the current one unchanges
                            skip_ga = True
                            generation_count = 0
                            if print_data:
                                print('MODEL LOST IN BAD THOUGHT POOL: RE-INIT GA PROCESSING')

                    if skip_ga == False:

                        new_population_len = []
                        generation_count += 1
                        global_generation_count += 1
                        if gentic_algorithm_active:

                            new_population = GenticAlgorithm.produceNextGeneration(population=agents.dead_agents,
                                                                          agent_meta_data=agents.__dict__)

                            if is_target_level and generation_count == target_generation:
                                    sum_fitness = 0

                                    for ag in agents.dead_agents:
                                       # ag.computeFitness()
                                        sum_fitness += ag.fitness

                                    avg_fitness = sum_fitness / (len(agents.dead_agents))
                                    global_generation_count = -1
                                    done = True
                                    break



                            agents.update_arrays(new_population)
                            new_population_len = len(new_population)

                        else:
                            agents = AgentManager(mode=MODE,
                                                  yPos_range=None,
                                                  starting_xPos=100,
                                                  starting_yPos=500,
                                                  xPos_range=None,
                                                  population_size=300,
                                                  screen_height=SCREEN_HEIGHT,
                                                  vertical_fuel_depletion_rate=0.001,
                                                  horizontal_fuel_depletion_rate=0.001,
                                                  screen_width=SCREEN_WIDTH,
                                                  rand_init_population_size=300,
                                                  rand_init_mutation_rate=0.0,
                                                  med_sim_init_population_size=300,
                                                  med_sim_init_rand_agent_percenatge=med_sim_init_rand_agent_percenatge,
                                                  med_sim_init_med_memory_agent_percenatge=med_sim_init_med_memory_agent_percenatge,
                                                  med_sim_init_rand_mutation_rate=med_sim_init_rand_mutation_rate,
                                                  med_sim_init_med_mutation_rate=med_sim_init_med_mutation_rate,
                                                  load_agent_filepath='sim_data/default_best'
                                                  )
                            new_population_len = len(agents.dead_agents)


                        if print_data:
                            print('generation = %s population size = %s level no = %s / %s' % (
                                generation_count, new_population_len, current_level_no, len(levels)))

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
                save_folder_path=save_folder_path, preview=True)

        if MODE == 'capture' and capture_success:
            if current_level_no < len(levels) - 1:
                current_level_no += 1
                current_level = levels[current_level_no]
            else:
                done = True


        pygame.display.flip()

        end_time = pygame.time.get_ticks()
        time = (end_time - start_time) / 1000
        clock.tick(60)

        if time >= run_time:
            done = True

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

            if print_data:
                if MODE == 'adapt':
                    print('Mode: %s is familiar: %s model in use: %s similarity: %s' % (MODE, is_familiar, 'Finding solution', sim_score))

                elif MODE == 'test':
                    print('Mode: %s is familiar: %s model in use: %s Similarity: %s' % (MODE, is_familiar, agents.not_sprites[0].functional_system.name,
                                                                                       sim_score))

            event_flag = False

    if print_data:
        print('Simulation Terminated')
        print('Simulation Time Length = %s' % start_time)
    pygame.quit()
    return  avg_fitness, global_generation_count

if __name__ == "__main__":
    run_model()