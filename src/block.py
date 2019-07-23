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

    def __init__(self, SCREEN_HEIGHT, SCREEN_WIDTH, class_label, width=100, speed=4):

        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.SCREEN_WIDTH = SCREEN_WIDTH

        self.width = width
        self.speed = speed
        self.xPos = SCREEN_WIDTH
        self.class_label = class_label
        self.configed_flag = False


    def rand_config(self, max_gap_size=80):

        topStart = int(np.round(np.random.rand() * (self.SCREEN_HEIGHT - max_gap_size)))
        min = topStart + max_gap_size
        bottomStart = int(np.floor(np.random.rand() * (self.SCREEN_HEIGHT - min + 1)) + min)


        self.topStart = topStart
        self.bottomStart = bottomStart
        self.configed_flag = True


    def manual_config(self, topStart, bottomStart):

        self.topStart = topStart
        self.bottomStart = bottomStart
        self.configed_flag = True

    def build(self):

        self.top_block = Component(width=self.width, height=self.topStart,
                                  xPos=self.xPos, yPos=0, color=self.BLACK,
                                  name="top_block_component")

        self.bottom_block = Component(width=self.width, height=self.SCREEN_HEIGHT-self.bottomStart,
                                  xPos=self.xPos, yPos=self.bottomStart, color=self.BLACK,
                                  name="top_block_component")


    def update(self):

        self.top_block.rect.x -= self.speed
        self.bottom_block.rect.x -= self.speed
        # keep a global record with xPos for both blocks
        self.xPos -= self.speed


    def hit(self, agent):

        if agent.rect.top  < self.top_block.rect.bottom or agent.rect.bottom > self.bottom_block.rect.top:
            if (agent.rect.midright[0] > self.xPos) and agent.rect.midright[0] < self.xPos + self.width:

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


if __name__ == "__main__":

    block = Block(SCREEN_HEIGHT=300, SCREEN_WIDTH=300)
    a = 0