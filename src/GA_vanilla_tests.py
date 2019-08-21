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

def run_simulation(DRAW, SCREEN_HEIGHT, SCREEN_WIDTH,
                   MODE, EXTRACT_DATA, SIM_SPEED,
                   save_first_winner, level_to_save,
                   generation_limit, end_on_lap,
                   levels, attempts):

    print('MODEL = *** GA VANILLA SEARCH ***')
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
                           med_sim_init_rand_agent_percenatge=0.0,
                           med_sim_init_med_memory_agent_percenatge=1.0,
                           med_sim_init_rand_mutation_rate=0.0,
                           med_sim_init_med_mutation_rate=0.2,
                           load_agent_filepath='sim_data/sim_test_main_agent',
                           save_path='sim_test_main_agent'
                         )


    screen, clock, current_level, current_level_no = init(SCREEN_WIDTH, SCREEN_HEIGHT, levels)

    internal_generation_count = 0
    level_generation_count = 0
    generation_record = np.zeros(len(levels)) - 1
    generation_record_count = 0
    attempts = attempts
    starting_attempts = attempts
    lap_no = 1
    change_level_flag = False
    winners = []

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



        for i in range(SIM_SPEED):

            # --- Update Agents ---
            i = 0
            while i < len(agents.not_sprites) and i > -1:
                agent = agents.not_sprites[i]

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

                print('LEVEL ' + str(current_level_no) + ' COMPLETE: GA VANILLA')
                attempts = starting_attempts

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
                internal_generation_count = 0

                if last_level:
                    if current_level_no == 0 and end_on_lap == lap_no:
                        done = True
                        break
                    else:
                        lap_no += 1




            # check if all active agents are dead, the perform GA and reset game level and epochs
            if len(agents.not_sprites) == 0:

                skip_ga = False
                if internal_generation_count >= generation_limit:

                    agents.dead_agents = []
                    agents = AgentManager(mode=MODE,
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
                                          med_sim_init_rand_agent_percenatge=1.0,
                                          med_sim_init_med_memory_agent_percenatge=0.0,
                                          med_sim_init_rand_mutation_rate=0.0,
                                          med_sim_init_med_mutation_rate=0.2,
                                          load_agent_filepath='sim_data/sim_test_main_agent',
                                          save_path='sim_test_main_agent'
                                          )  # leave one model the current one unchanges
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


                    print('MODEL LOST IN BAD THOUGHT POOL: ATTEMPTS = %s out %s RE-INIT GA PROCESSING' % (attempts, starting_attempts))

                if skip_ga == False:

                    new_population = GenticAlgorithm.produceNextGeneration(population=agents.dead_agents,
                                                                  agent_meta_data=agents.__dict__)

                    agents.update_arrays(new_population)

                    internal_generation_count += 1
                    print('generation = %s level_generation = %s population size = %s level no = %s / %s' % (
                        internal_generation_count, level_generation_count, len(new_population), current_level_no, len(levels)))

                    # if MODE != 'adapt':
                    #     current_level_no = 0
                    #     current_level = levels[current_level_no]



            # --- Drawing ---
            screen.fill(WHITE)

            if DRAW:
                for block in current_level:
                    block.draw(screen)

                agents.draw(screen, mode=MODE)

            pygame.display.flip()

            clock.tick()
            frame_count += 1
            if change_level_flag:
                change_level_flag = False


    pygame.quit()
    print('SIMULATION LEVEL COMPLETED BY AGENT')
    return generation_record

if __name__ == "__run_simulation__":
    #gen_rec = run_simulation()
    pass