import numpy as np
import pygame


class Component(pygame.sprite.Sprite):

    def __init__(self, width, height, xPos, yPos, color, name):

        super().__init__()
        self.name = name
        self.image = pygame.Surface([width, height])
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.x = xPos
        self.rect.y = yPos


class Block():

    def __init__(self, SCREEN_HEIGHT, SCREEN_WIDTH, width=100, color=(0,0,0), class_label=''):

        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.color = color
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.SCREEN_WIDTH = SCREEN_WIDTH

        self.width = width
        self.xPos = SCREEN_WIDTH
        self.class_label = class_label
        self.configed_flag = False


    def rand_config(self,xPos,  max_gap_size=80):

        topStart = int(np.round(np.random.rand() * (self.SCREEN_HEIGHT - max_gap_size)))
        min = topStart + max_gap_size
        bottomStart = int(np.floor(np.random.rand() * (self.SCREEN_HEIGHT - min + 1)) + min)

        self.xPos = xPos
        self.topStart = topStart
        self.bottomStart = bottomStart
        self.configed_flag = True


    def manual_config(self, xPos, topStart, bottomStart):

        self.xPos = xPos
        self.topStart = topStart
        self.bottomStart = bottomStart
        self.configed_flag = True


    def build(self):

        self.top_block = Component(width=self.width, height=self.topStart,
                                  xPos=self.xPos, yPos=0, color=self.color,
                                  name="top_block_component")

        self.bottom_block = Component(width=self.width, height=self.SCREEN_HEIGHT-self.bottomStart,
                                  xPos=self.xPos, yPos=self.bottomStart, color=self.color,
                                  name="top_block_component")


    def hit(self, agent):

        if agent.rect.top  < self.top_block.rect.bottom or agent.rect.bottom > self.bottom_block.rect.top:
            if agent.rect.right > self.bottom_block.rect.left and agent.rect.right < self.bottom_block.rect.right:

                return True

        return False


    def offscreen(self):

        if self.xPos < -self.width:
            return True
        else:
            return False


    def draw(self, surface):

        draw_group = pygame.sprite.Group()
        draw_group.add(self.top_block)
        draw_group.add(self.bottom_block)

        draw_group.draw(surface)

    @staticmethod
    def generate(screen_height, screen_width, max_gap_size=100, split_first_args=False, *args):

        level_list = []

        if split_first_args and len(args) == 1:
            d = []
            for item in args[0]:
                d.append(item)

            args = d

        for level in args:
            block_list = []
            for block in level:
                b = Block(SCREEN_HEIGHT=screen_height, SCREEN_WIDTH=screen_width,
                          width=block[3], color=block[4], class_label='block')

                if block[0] == 'random':
                    b.rand_config(max_gap_size=max_gap_size)

                elif isinstance(block[0], int) and isinstance(block[1], int) and isinstance(block[2], int) :
                    b.manual_config(xPos=block[0], topStart=block[1], bottomStart=block[2])

                else:
                    raise ValueError('ERROR: Block blueprint does not contain '
                                     'a valid integer entry or a string entry names random')

                b.build()
                block_list.append(b)

            level_list.append(block_list)
        return level_list


if __name__ == "__main__":


    L1 =     [
             [390, 50, 20, 500, (0, 0, 255)]
             ]

    L2 =     [
             [190, 50, 20, 500, (0, 0, 255)],
             [590, 50, 20, 500, (0, 0, 255)]
             ]

    L3 =     [[0, 0, 20, 250, (0, 0, 255)],
             [0, 350, 20, 250, (0, 0, 255)],
             [780, 0, 20, 250, (0, 0, 255)],
             [780, 350, 20, 250, (0, 0, 255)],
             [20, 0, 760, 20, (0, 0, 255)],
             [20, 580, 760, 20, (0, 0, 255)]
             ]

    levels = Block.generate(800, 600, 100, L1, L2, L3)
    l = 1
    # block = Block(SCREEN_HEIGHT=800, SCREEN_WIDTH=600, width=100, class_label='block')
    # block.manual_config(400, 460)
    # block.build()
    # a= 1