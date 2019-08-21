import pygame
import numpy as np
from nn import NeuralNetwork
import math


class Agent(pygame.sprite.Sprite):

    def __init__(self, xPos, yPos,xPos_range, yPos_range, initEmpty,
                 vertical_fuel_depletion_rate=0.05, horizontal_fuel_depletion_rate=0.05,
                 name='agent', color=(0, 0, 0, 50)):
        # Call the parent's constructor
        super().__init__()

        self.gravity  = 0.0
        self.drag     = 0.0
        self.lift     = -10
        self.push     = 2
        self.maxLim_y_velocity   = 20
        self.minLim_y_velocity   = -20
        self.maxLim_x_velocity   = 4
        self.minLim_x_velocity   = -4
        self.velocity_y = 0
        self.velocity_x = 0
        self.radius   = 20
        self.color = color
        self.current_closest_block = None
        self.fuel = 1.0
        self.failure_meter = 0.0
        self.vertical_fuel_depletion_rate = vertical_fuel_depletion_rate
        self.horizontal_fuel_depletion_rate = horizontal_fuel_depletion_rate

        if xPos_range is not None:
            xPos = np.random.randint(xPos_range[0], xPos_range[1])

        if yPos_range is not None:
            yPos = np.random.randint(yPos_range[0], yPos_range[1])


        self.name = name
        self.image = pygame.Surface([self.radius, self.radius], pygame.SRCALPHA)
        self.image.fill(self.color)
        self.rect = self.image.get_rect()
        self.rect.x = xPos
        self.rect.y = yPos
        self.previous_xPos = self.rect.right
        self.starting_xPos = xPos
        self.starting_yPos = yPos

        self.timeSamplesExperianced       = 1
        self.totalDistanceFromGapOverTime = 0

        self.fitness        = 0
        self.avgDistFromGap = 0


        msLayeruUnits = [12, 7, 2]
        msActFunctions = ["relu", "tanh"]

        self.functional_system = NeuralNetwork(layer_units=msLayeruUnits, activation_func_list=msActFunctions)

        if initEmpty == False:
            self.functional_system.init_layers(init_type="he_normal")

        else:
            self.functional_system.init_layers(init_type="zeros")

    def reset(self):

        self.velocity_y = 0
        self.current_closest_block = None
        self.fuel = 1.0

        self.rect.x = self.starting_xPos
        self.rect.y = self.starting_yPos
        self.previous_xPos = self.rect.center[0]

        self.timeSamplesExperianced       = 1
        self.totalDistanceFromGapOverTime = 0

        self.fitness        = 0
        self.avgDistFromGap = 0



    def think(self, active_blocks, screen_width, screen_height):

        # find closest block for data input

        result = list(filter(lambda x: (x.top_block.rect.right > self.rect.right), active_blocks))
        result.sort(key=lambda x:(x.top_block.rect.right))

        if len(result) != 0:

            closest_block = result[0]
            inputs = []
            #input about on coming block object
            inputs.append(closest_block.xPos / screen_width)
            inputs.append(closest_block.top_block.rect.bottom / screen_height)
            inputs.append(closest_block.bottom_block.rect.top / screen_height)
            inputs.append((closest_block.xPos - self.rect.right) / screen_width)

            # input about agents current position
            inputs.append((screen_height - self.rect.bottom)/screen_height) # distance from bottom of screen
            inputs.append(self.rect.bottom / screen_height)
            inputs.append(self.rect.top / screen_height)
            inputs.append(self.rect.right / screen_height)
            inputs.append(self.rect.left / screen_height)
            inputs.append(self.fuel)
           # inputs.append(self.vertical_fuel_depletion_rate)
           # inputs.append(self.horizontal_fuel_depletion_rate)
            inputs.append(self.minMaxNormalise(self.velocity_x, min=self.minLim_x_velocity, max=self.maxLim_x_velocity))
            inputs.append(self.minMaxNormalise(self.velocity_y, min=self.minLim_y_velocity, max=self.maxLim_y_velocity))

            inputs = np.array([inputs])

            prediction = self.functional_system.feed_foward(inputs=inputs)


            self.actionVertical(input=prediction[0], mode='joystick_control')


            self.actionHorizontal(input=prediction[1], mode='joystick_control')
            # if prediction[0] > prediction[1]:
            #     self.actionVertical()

            # if prediction[2] > prediction[3]:
            #     self.actionHorizontal()


            self.current_closest_block = closest_block


    def actionVertical(self, input=None, mode='discrete_control'):

        if mode == 'discrete_control':
            self.velocity_y += self.lift
        elif mode == 'joystick_control':
            self.velocity_y += self.lift * input
        else:
            raise ValueError('ERROR: Invalid action control entry')

        self.fuel -= self.vertical_fuel_depletion_rate
        self.color_gauge(self.vertical_fuel_depletion_rate)


    def actionHorizontal(self, input=None, mode='discrete_control'):

        if mode == 'discrete_control':
            self.velocity_x += self.push
        elif mode == 'joystick_control':
            self.velocity_x += self.push * input
        else:
            raise ValueError('ERROR: Invalid action control entry')

        self.fuel -= self.horizontal_fuel_depletion_rate
        self.color_gauge(self.horizontal_fuel_depletion_rate)


    def color_gauge(self, deduction):
        self.failure_meter += deduction
        if self.failure_meter > 1: self.failure_meter = 0
        c = list(self.color)
        c[0] = self.failure_meter*255
        self.color = tuple(c)
        self.image.fill(self.color)


    def update(self, screen_height):

        self.velocity_y += self.gravity
        self.velocity_y *= 0.9
        self.rect.y += self.velocity_y


        self.velocity_x += self.drag
        # self.velocity_x *= 0.9
        self.rect.x += self.velocity_x

        if self.velocity_y > self.maxLim_y_velocity:
            self.velocity_y = self.maxLim_y_velocity

        if self.velocity_y < self.minLim_y_velocity:
            self.velocity_y = self.minLim_y_velocity

        if self.velocity_x > self.maxLim_x_velocity:
            self.velocity_x = self.maxLim_x_velocity

        if self.velocity_x < self.minLim_x_velocity:
            self.velocity_x = self.minLim_x_velocity


        # penalise agents for their distance on the y from the center of the gap of the blocks
        gap = self.current_closest_block.bottom_block.rect.top - self.current_closest_block.top_block.rect.bottom
        gapMid = self.current_closest_block.top_block.rect.bottom + np.round((gap / 2))
        agentDistanceFromGap = np.floor(np.abs(self.rect.midright[1] - gapMid))

        self.totalDistanceFromGapOverTime += agentDistanceFromGap
        self.timeSamplesExperianced += 1

        if self.rect.right > self.previous_xPos:
            # fitness only increses if the agent is moving to the right towards the goal
            self.fitness += 1

        self.previous_xPos = self.rect.right


    def fuel_depleted(self):
        if self.fuel < 0:
            return True

        return False

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

    def minMaxNormalise(self, x, min, max):
        return (x - min) / (max - min)


    def computeFitness(self):

        self.fitness = math.pow(self.fitness, 4)

        impactFactor = 0.5 # scales the percentage of penalisation applied
        self.avgDistFromGap = np.floor(self.totalDistanceFromGapOverTime / self.timeSamplesExperianced)
        fitness_penalty =np.floor(impactFactor * self.avgDistFromGap)
        self.fitness -= fitness_penalty
        if self.fitness < 0:
            self.fitness = 0





if __name__ == "__main__":
    pass




