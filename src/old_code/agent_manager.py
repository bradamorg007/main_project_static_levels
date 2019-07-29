import pygame
from old_code.agent import Agent
import pickle as pkl

class AgentManager:
    def __init__(self, population_size, level_manager):
        not_sprites = []
        load_flag = False
        for i in range(population_size):
            agent = Agent(initBrain=False, initEmpty=False,
                                           screen_width=level_manager.SCREEN_WIDTH,
                                           screen_height=level_manager.SCREEN_HEIGHT, name=str(i))

            if level_manager.mode == 'test' and level_manager.load_agent_filepath is not None:
                file = open(level_manager.load_agent_filepath, 'rb')
                loaded_functional_system = pkl.load(file)
                file.close()

                agent.functional_system = loaded_functional_system
                load_flag = True
                print('NOTIFICATION: Pre-designed Agent functional System has been loaded. Population size will be ignored')



            not_sprites.append(agent)

            self.len = len(not_sprites)
            self.sprites = pygame.sprite.Group()
            self.not_sprites = not_sprites
            self.dead_agents = []

            if load_flag:
                break


    def splice(self, index):

        dead_agent = self.not_sprites.pop(index)
        self.dead_agents.append(dead_agent)
        self.len = len(self.not_sprites)


    def draw(self, surface, mode):

        if mode == 'train' or mode == 'test':
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


if __name__ == '__main__':
    pass