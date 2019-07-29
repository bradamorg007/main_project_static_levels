import pygame
from agent import Agent
import pickle as pkl
import numpy as np
import os

class AgentManager:
    def __init__(self, population_size, screen_height, screen_width,
                 mode,  vertical_fuel_depletion_rate=0.05, horizontal_fuel_depletion_rate=0.05,
                 starting_xPos=20, starting_yPos=100,xPos_range=None, yPos_range=None,
                 save_path='default_best', load_agent_filepath=None):

        self.vertical_fuel_depletion_rate = vertical_fuel_depletion_rate
        self.horizontal_fuel_depletion_rate = horizontal_fuel_depletion_rate
        self.starting_xPos = starting_xPos
        self.starting_yPos = starting_yPos
        self.xPos_range = xPos_range
        self.yPos_range = yPos_range
        self.save_path = save_path
        self.temp_agent_store = []
        self.temp_functional_system_store = None

        self.not_sprites = []
        self.dead_agents = []
        load_flag = False


        for i in range(population_size):


            agent = Agent(
                          vertical_fuel_depletion_rate=vertical_fuel_depletion_rate,
                          horizontal_fuel_depletion_rate=horizontal_fuel_depletion_rate,
                          xPos=starting_xPos, yPos=starting_yPos, xPos_range=xPos_range,
                          yPos_range=yPos_range,
                          initEmpty=False, name=str(i)
                         )

            if mode == 'test' and load_agent_filepath is not None:
                file = open(load_agent_filepath, 'rb')
                loaded_functional_system = pkl.load(file)
                file.close()

                agent.functional_system = loaded_functional_system
                load_flag = True
                print('NOTIFICATION: Pre-designed Agent functional System has been loaded. Population size will be ignored')


            self.not_sprites.append(agent)

            if load_flag:
                break

        self.len = len(self.not_sprites)
        self.sprites = pygame.sprite.Group()


    def reset_temp_agent_store(self, functional_system= None):

        if functional_system is None:
            self.not_sprites = self.temp_agent_store
            self.temp_agent_store = []

        else:
            self.temp_functional_system_store = self.temp_agent_store[0].functional_system
            self.temp_agent_store[0].functional_system = functional_system
            self.not_sprites = self.temp_agent_store
            self.temp_agent_store = []


    def restore_original_test_agent(self):
        if self.temp_functional_system_store is not None:
            print('RESTORING ORIGINAL SOLUTION FOR FAMILIAR ENVIRONMENT')
            self.not_sprites[0].functional_system = self.temp_functional_system_store

        self.temp_functional_system_store = None


    def store_test_agent_progress(self, mode):

        if mode == 'adapt':

            if len(self.not_sprites) == 0 and len(self.temp_agent_store) == 1:
                return self.temp_agent_store[0]
            else:
                current_agent = self.not_sprites[0]
                self.temp_agent_store.append(current_agent)
                return current_agent



    def adaptive_rand_population_init(self, mode, population_size, rand_to_current_model_split=(0.5,0.5),
                                      rand_model_mutation_rate=0.1, current_model_mutation_rate=0.1):

        current_model_template = self.store_test_agent_progress(mode)
        xPos = current_model_template.rect.x
        yPos = current_model_template.rect.y
        current_agent_functional_system = current_model_template.functional_system
        self.not_sprites = []

        rand_model_amount = int(np.round(population_size * rand_to_current_model_split[0]))
        current_model_amount = int(np.round(population_size * rand_to_current_model_split[1]))

        for i in range(rand_model_amount):
            agent = Agent(
                vertical_fuel_depletion_rate=self.vertical_fuel_depletion_rate,
                horizontal_fuel_depletion_rate=self.horizontal_fuel_depletion_rate,
                xPos=xPos, yPos=yPos, xPos_range=None,
                yPos_range=None,
                initEmpty=False, name='adapted_network'+str(i), color=(0, 255, 255, 100)
            )
            agent.functional_system.name = 'adapted_network'+str(i)

            agent.functional_system.mutate(rate=rand_model_mutation_rate)
            self.not_sprites.append(agent)

        for i in range(current_model_amount):
            agent = Agent(
                vertical_fuel_depletion_rate=self.vertical_fuel_depletion_rate,
                horizontal_fuel_depletion_rate=self.horizontal_fuel_depletion_rate,
                xPos=xPos, yPos=yPos, xPos_range=None,
                yPos_range=None,
                initEmpty=False, name='adapted_network'+str(i), color=(0, 0, 255, 100)
            )
            agent.functional_system.name = 'adapted_network' + str(i)


            agent.functional_system = current_agent_functional_system
            agent.functional_system.mutate(rate=current_model_mutation_rate)
            self.not_sprites.append(agent)





    def splice(self, index, mode=''):

        if len(self.not_sprites) != 0:
            if  mode == 'all':
                for agent in self.not_sprites:
                    self.dead_agents.append(agent)
                    self.not_sprites = []

            elif mode == 'delete':
                self.not_sprites.pop(index)


            elif isinstance(index, int) and mode=='':
                dead_agent = self.not_sprites.pop(index)
                self.dead_agents.append(dead_agent)

            self.len = len(self.not_sprites)


    def draw(self, surface, mode):

        if mode == 'train' or mode == 'test' or mode == 'adapt':
            self.sprites = pygame.sprite.Group()
            self.sprites.add(self.not_sprites)
            self.sprites.draw(surface)


    def clear(self, mode):
        if mode == 'train':
          self.sprites = pygame.sprite.Group()


    def update_arrays(self, input):

        if isinstance(input, list):

            self.not_sprites = input
            self.sprites = pygame.sprite.Group()
            self.len = len(self.not_sprites)
            self.dead_agents = []

        else:
            raise ValueError("ERROR ActiveAgents: input must be of type list")

    def save_best_agent(self):

        # save best agent
        max = 0
        index = 0
        for i, agent in enumerate(self.not_sprites):
            agent.computeFitness()
            if agent.fitness > max:
                max = agent.fitness
                index = i
        self.not_sprites[index].functional_system.name = 'original_network'
        folder = "sim_data"
        filename = os.path.join(folder, self.save_path)
        if os.path.isdir(folder) == False:
            os.mkdir(folder)

        pkl.dump(self.not_sprites[index].functional_system, open(os.path.join(filename), 'wb'))
        print('NOTIFICATION: Best Agent has been saved to %s' % filename)
        print('NOTIFICATION: simulation environment has been saved to %s' % 'sim_env.pkl')


if __name__ == '__main__':
    pass