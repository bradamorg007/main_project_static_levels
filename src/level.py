import pygame
import numpy as np


class Wall(pygame.sprite.Sprite):
    """This class represents the bar at the bottom that the player controls """

    def __init__(self, x, y, width, height, color):
        """ Constructor function """

        # Call the parent's constructor
        super().__init__()

        # Make a BLUE wall, of the size specified in the parameters
        self.image = pygame.Surface([width, height])
        self.image.fill(color)

        # Make our top-left corner the passed-in location.
        self.rect = self.image.get_rect()
        self.rect.y = y
        self.rect.x = x



class Level(object):
    """ Base class for all rooms. """

    # Each room has a list of walls, and of enemy sprites.
    wall_list = None
    enemy_sprites = None

    def __init__(self, wall_blueprints, max_width=None, max_height=None):



        """ Constructor, create our lists. """
        self.wall_list = pygame.sprite.Group()
        self.enemy_sprites = pygame.sprite.Group()

        # Loop through the list. Create the wall, add it to the list
        for item in wall_blueprints:

            if item == 'random':
                item = [np.random.randint(max_width),
                        np.random.randint(max_height),
                        np.random.randint(max_width),
                        np.random.randint(max_height),
                        (0,0,0)]

            wall = Wall(item[0], item[1], item[2], item[3], item[4])
            self.wall_list.add(wall)



# # EXAMPLE
#         # This is a list of walls. Each is in the form [x, y, width, height]
#         walls = [[0, 0, 20, 250, WHITE],
#                  [0, 350, 20, 250, WHITE],
#                  [780, 0, 20, 250, WHITE],
#                  [780, 350, 20, 250, WHITE],
#                  [20, 0, 760, 20, WHITE],
#                  [20, 580, 760, 20, WHITE],
#                  [390, 50, 20, 500, BLUE]
#                  ]

if __name__ == '__main__':

    WHITE = (255, 255, 255, 255)
    walls = [[0, 0, 20, 250, WHITE],
                     [0, 350, 20, 250, WHITE],
                     [780, 0, 20, 250, WHITE],
                     [780, 350, 20, 250, WHITE],
                     [20, 0, 760, 20, WHITE],
                     [20, 580, 760, 20, WHITE],
                     [390, 50, 20, 500, WHITE]
                     ]



    l = Level(wall_blueprints=walls)