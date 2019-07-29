import pygame
import numpy as np
from nn import NeuralNetwork


class Agent(pygame.sprite.Sprite):

    def __init__(self, initBrain, initEmpty,
                 screen_width, screen_height,
                 name='agent'):
        # Call the parent's constructor
        super().__init__()

        self.gravity  = 0.6
        self.drag     = 0.6
        self.lift     = -15
        self.push     = 15
        self.maxLim   = 6
        self.minLim   = -15
        self.velocity_y = 0
        self.velocity_x = 0
        self.radius   = 25
        self.color = (0, 0, 0, 50)

        self.name = name
        self.image = pygame.Surface([self.radius, self.radius], pygame.SRCALPHA)
        self.image.fill(self.color)
        self.rect = self.image.get_rect()
        self.rect.x = 20
        self.rect.y = screen_height/2

        self.timeSamplesExperianced       = 0
        self.totalDistanceFromGapOverTime = 0

        self.fitness        = 0
        self.avgDistFromGap = 0


        msLayeruUnits = [10, 7, 4]
        msActFunctions = ["relu", "softmax"]

        self.functional_system = NeuralNetwork(layer_units=msLayeruUnits, activation_func_list=msActFunctions)

        if initEmpty == False:
            self.functional_system.init_layers(init_type="he_normal")

        else:
            self.functional_system.init_layers(init_type="zeros")


    def show(self):
        pass


    def think(self, closestBlock, screen_width, screen_height):

        inputs = []
        # input about on coming block object
        inputs.append(closestBlock.xPos / screen_width)
        inputs.append(closestBlock.top_block.rect.bottom / screen_height)
        inputs.append(closestBlock.bottom_block.rect.top / screen_height)
        inputs.append((closestBlock.xPos - self.rect.right) / screen_width)

        # input about agents current position
        inputs.append((screen_height - self.rect.bottom)/screen_height) # distance from bottom of screen
        inputs.append(self.rect.bottom / screen_height)
        inputs.append(self.rect.top / screen_height)
        inputs.append(self.rect.right / screen_height)
        inputs.append(self.rect.left / screen_height)
        inputs.append(self.minMaxNormalise(self.velocity_y))

        inputs = np.array([inputs])

        prediction = self.functional_system.feed_foward(inputs=inputs)

        if prediction[0] > prediction[1]:
            self.actionUp()

        if prediction[2] > prediction[3]:
            self.actionHorizontal()


    def actionUp(self):
        self.velocity_y += self.lift

    def actionHorizontal(self):
        self.velocity_x += self.push


    def reset(self, screen_height):
        self.gravity  = 0.6
        self.drag     = 0.6
        self.lift     = -15
        self.push     = 5
        self.maxLim   = 6
        self.minLim   = -15
        self.velocity_y = 0
        self.velocity_x = 0
        self.radius   = 25
        self.color = (0, 0, 0, 50)

        self.image = pygame.Surface([self.radius, self.radius], pygame.SRCALPHA)
        self.image.fill(self.color)
        self.rect = self.image.get_rect()
        self.rect.x = 10
        self.rect.y = screen_height/2

        self.timeSamplesExperianced       = 0
        self.totalDistanceFromGapOverTime = 0

        self.fitness        = 0
        self.avgDistFromGap = 0

    def update(self, closestBlock, screen_height):

        self.velocity_y += self.gravity
        self.velocity_y *= 0.9
        self.rect.y += self.velocity_y

        self.velocity_x += self.drag
        self.velocity_x *= 0.1
        self.rect.x += self.velocity_x

        if self.velocity_y > self.maxLim:
            self.velocity_y = self.maxLim

        if self.velocity_y < self.minLim:
            self.velocity_y = self.minLim

        # if self.rect.bottom > screen_height:
        #     self.rect.bottom = screen_height
        #     self.velocity = 0
        #
        # elif self.rect.top < 5:
        #     self.rect.top = 0
        #     self.velocity = 0

        # penalise agents for their distance on the y from the center of the gap of the blocks
        gap = closestBlock.bottom_block.rect.top - closestBlock.top_block.rect.bottom
        gapMid = closestBlock.top_block.rect.bottom + np.round((gap / 2))
        agentDistanceFromGap = np.floor(np.abs(self.rect.midright[1] - gapMid))

        self.totalDistanceFromGapOverTime = self.totalDistanceFromGapOverTime + agentDistanceFromGap
        self.timeSamplesExperianced = self.timeSamplesExperianced + 1

        self.fitness = self.fitness + 1

    def off_screen(self, screen_height, screen_width):

        if self.rect.top < 5:
            return True
        elif self.rect.bottom > screen_height:
            return True
        elif self.rect.left < 0:
            return True
        elif self.rect.right > screen_width:
            return True
        else:
            return False

    def minMaxNormalise(self, x):
        return (x - self.minLim) / (self.maxLim - self.minLim)


    def computeFitness(self):
        # penalise agent based on average distance from gap

        impactFactor = 0.9 # scales the percentage of penalisation applied
        self.avgDistFromGap = np.floor(self.totalDistanceFromGapOverTime / self.timeSamplesExperianced)
        self.fitness = self.fitness - np.floor(impactFactor * self.avgDistFromGap)
        if self.fitness < 0:
            self.fitness = 0


if __name__ == "__main__":
    pass



