import math
import numpy as np
from agent import Agent
from Random import Random

class GenticAlgorithm:

    mutateRate = 0.1
    selectionType = "roullete"
    crossOverRate = 0.5



    @staticmethod
    def produceNextGeneration(population, agent_meta_data):

        # // To produce the next generation several steps are required
        #
        # // NOTE
        # // GA Could select Parent A and B from the same neural network if its very high fitness
        # // So the GA may favor one particular network, but mutation can still happen to keep things
        # // changing
        #
        # // 1.) sum the fitness and add exponetial curvature to fitness
        # // We add exponential curvature to greater distinctions between the performance of agents
        # // e.g A fitness = 20, B fitness = 19 here the fitness difference is very small only one.
        # // both A and B will have a very simular probability of been selected even tho A is better
        # // but 20^2 = 400 and 19^2 = 361 this creates a muc bigger difference between their performances
        # // now A is much more likely to be selected than B
        #
        # // Order Population from largest fitness to smallest. larger fitness are more likely to be selected
        # // so we might aswell iterate through them first, allows use to break out of the list

        vert_depletion = agent_meta_data.get('vertical_fuel_depletion_rate')
        hori_depletion = agent_meta_data.get('horizontal_fuel_depletion_rate')
        xPos = agent_meta_data.get('starting_xPos')
        yPos = agent_meta_data.get('starting_yPos')
        x_pos_range = agent_meta_data.get('xPos_range')
        y_pos_range = agent_meta_data.get('yPos_range')

        agent_meta_data = [vert_depletion, hori_depletion, xPos, yPos, x_pos_range, y_pos_range]
        newPopulation = []
        fitnessSum = 0

        # FITNESS FUNCTION: uses power fo non_linear Fitness
        for i in range(len(population)):
            #population[i].fitness = math.pow(population[i].fitness, 4)
            population[i].computeFitness()
            fitnessSum = fitnessSum + population[i].fitness

        # // // 2.) Proportional fitness probabilities Normalise the agent fitnesss now that we have the sum.
        # // for (let i = 0; i < population.length; i++) {
        # //     population[i].fitness = population[i].fitness / fitnessSum;
        # // }
        #
        # // 3.) now I need to create a new population of children

        # i throw two darts to choose Parent A and B
        # parent A gets chosen its fitness is set 0 afterwards to reduce the chance of it been selected as Parent B
        # want to avoid same poarent breeding as it will destroy diversity
        parentA_fitness = None
        index = None
        for i in range(len(population)):

            if i > 0:
                population[index].fitness = parentA_fitness

            parentA, index = GenticAlgorithm.selectParent(population, fitnessSum)
            parentA_fitness = parentA.fitness
            population[index].fitness = 0

            parentB, _, = GenticAlgorithm.selectParent(population, fitnessSum)
            newPopulation.append(GenticAlgorithm.reproduceAndMutate(parentA, parentB,agent_meta_data))

        return newPopulation

    @staticmethod
    def selectParent(population, fitnessSum):

        index = 0
        rand_select = False
        r = np.round(np.random.rand() * fitnessSum)

        # if index is larger than len-1 then it means all agents have the same fitness probs 0
        # if this is the case then all one can do is just randomly select one. will happen in
        # the case hwre only one agent got a fitness greater than 0 but it can only be selected once after which its
        # is set to 0 to allow other agentrs to be selected to avoid same parent breeding

        while r > 0:
            if index > len(population)-1:
                # random select
                rand_select = True
                #print('NOTIFICATION GA: All agents have the same fitness init random selection')
                break

            r = r - population[index].fitness

            if r > 0:
                index = index + 1

        if rand_select:
            index = np.random.randint(0, len(population)-1)

        parent = population[index]

        if parent == None:
            raise ValueError("ERROR GA: Parent in select parent method is undefined this is due to the indexing")

        return parent, index

    @staticmethod
    def reproduceAndMutate(parentA, parentB, AMD):
        # // Now go through Parents parmaters and exchange gentic info
        # //  Also mutate select gene within the same loop
        # // no need having a seperte mutate function that loops through paramter matrices again
        #
        # // Loops can use child dimensions as all networks have fixed same topologies in this


        child = Agent(initEmpty=True,
                      xPos=AMD[2], yPos=AMD[3], xPos_range=AMD[4], yPos_range=AMD[5],
                      vertical_fuel_depletion_rate=AMD[0],
                      horizontal_fuel_depletion_rate=AMD[1], color=parentA.color)
        child.functional_system.name = parentA.functional_system.name


        for i in range(len(child.functional_system.layers)):

            rowsW = child.functional_system.layers[i]['weights'].shape[0]
            colsW = child.functional_system.layers[i]['weights'].shape[1]

            for j in range(rowsW):
                for k in range(colsW):

                    if np.random.rand() < GenticAlgorithm.crossOverRate:
                        # Use Parent A gene
                        child.functional_system.layers[i]['weights'][j][k] = parentA.functional_system.layers[i]['weights'][j][k]

                    else:
                        child.functional_system.layers[i]['weights'][j][k] = parentB.functional_system.layers[i]['weights'][j][k]


                    if np.random.rand() < GenticAlgorithm.mutateRate:
                        child.functional_system.layers[i]['weights'][j][k] += Random.gaussian_distribution(mean=0, sigma=0, samples=1)

            # Reproduce and Mutate Baiases
            rowsB = child.functional_system.layers[i]['biases'].shape[0]
            colsB = child.functional_system.layers[i]['biases'].shape[1]

            for j in range(rowsB):
                for k in range(colsB):

                    if np.random.rand() < GenticAlgorithm.crossOverRate:
                        # Use Parent A gene
                        child.functional_system.layers[i]['biases'][j][k] = parentA.functional_system.layers[i]['biases'][j][k]

                    else:
                        child.functional_system.layers[i]['biases'][j][k] = parentB.functional_system.layers[i]['biases'][j][k]

                    if np.random.rand() < GenticAlgorithm.mutateRate:
                        child.functional_system.layers[i]['biases'][j][k] += Random.gaussian_distribution(mean=0, sigma=0,
                                                                                                       samples=1)
        return child



if __name__ == "__main__":

    parentA = Agent(initBrain=False, initEmpty=False, screen_width=800, screen_height=300)
    parentB = Agent(initBrain=False, initEmpty=False, screen_width=800, screen_height=300)
    child = GenticAlgorithm.reproduceAndMutate(parentA, parentB, screen_width=800, screen_height=300)