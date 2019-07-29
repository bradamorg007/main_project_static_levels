
import numpy as np
import math

class MemorySystem:

    def __init__(self, low_simularity_threshold,
                       high_simularity_threshold,
                       forget_usage_threshold,
                       forget_age_threshold,
                       max_memory_size):

        self.memories = []
        self.highly_similarity_memories = []
        self.medium_similarity_memories = []
        self.low_similarity_memories = []

        self.low_simularity_threshold    = low_simularity_threshold
        self.high_simularity_threshold   = high_simularity_threshold

        self.forget_usage_threshold      = forget_usage_threshold
        self.forget_age_threshold        = forget_age_threshold

        self.max_memory_size             = max_memory_size
        self.current_action = ''



    def create_memory(self, latent_representation, solution, tag=''):

        # should forget stuff come before making new memeories

        if len(self.memories) < self.max_memory_size:

            memory = {'latent_representation': latent_representation,
                      'solution': solution, 'usage': 0, 'age': 0,
                      'tag': tag, 'similarity_score': 0}

            self.memories.append(memory)

        # Need an else statement with rules what to do when memory gets full
        # need a mechanism to forget. maybe remove memories that are old so high age to some threshold and have a low
        # usage to some threshold. This means that the memory has been in the system for a while and has not been used
        # frequently thus it maybe worth forgetting this information

    def clean_up(self):
        self.highly_similarity_memories = []
        self.low_similarity_memories = []

    def query(self, current_latent_representation):

        # first find the smallest simularity value

        i = 0
        while i < len(self.memories):

            memory =  self.memories[i]
            memory_latent_representations = memory.get('latent_representation')
            simularity = self.euclidean_distance(current_latent_representation, memory_latent_representations)

            if simularity <= self.high_simularity_threshold:
                # high simularity
                # does not require adaptation
                self.memories[i]['tag'] = 'hs'
                memory['similarity_score'] = simularity
                self.highly_similarity_memories.append(memory)

            elif simularity <= self.low_simularity_threshold and simularity > self.high_simularity_threshold:
                # medium simularity
                # requires adaptation with memories as reference
                self.memories[i]['tag'] = 'ms'
                memory['similarity_score'] = simularity
                self.medium_similarity_memories.append(memory)

            elif simularity > self.low_simularity_threshold:
                # low simularity
                # requires adaptation heavily reliant on random inits
                self.memories[i]['tag'] = 'ls'
                memory['similarity_score'] = simularity
                self.low_similarity_memories.append(memory)

            i += 1

        memory = None
        action = None
        if len(self.highly_similarity_memories) > 0:
            min = self.highly_similarity_memories[0].get('similarity_score')
            index = 0

            for i, mem in enumerate(self.highly_similarity_memories):
                if mem.get('similarity_score') < min:
                    min = mem.get('similarity_score')
                    index = i

            memory = self.highly_similarity_memories[index]
            action = 'memory_to_fs_system_switch'
            self.current_action = action

        elif len(self.highly_similarity_memories) == 0 and len(self.medium_similarity_memories) > 0:
            memory = None
            action = 'adaption_using_medium_memory_as_init_foundation'
            self.current_action = action

        elif len(self.highly_similarity_memories) == 0 and len(self.medium_similarity_memories) == 0 and len(self.low_similarity_memories) > 0:
            memory = None
            action = 'adaption_using_low_memory_and_random_init_foundation'
            self.current_action = action

        elif len(self.memories) == 0:
            memory = None
            action = 'adaption_using_low_memory_and_random_init_foundation'
            self.current_action = action

        self.clean_up()
        return memory, action




    def euclidean_distance(self, vectorA, vectorB):

        if len(vectorA) != len(vectorB):
            raise ValueError('ERROR MEMORY SYSTEM EUCLIDEAN DISTANCE: '
                             'Input vectors must be of equal length')

        sum = 0
        for pointA, pointB in zip(vectorA, vectorB):
            diff = pointA - pointB
            sum += math.pow(diff, 2)

        result = math.sqrt(sum)
        return result

    @staticmethod
    def init(MODE, low_simularity_threshold,high_simularity_threshold,
                               forget_usage_threshold,forget_age_threshold,
                               max_memory_size):

        if MODE == 'test':
            ms = MemorySystem(low_simularity_threshold=low_simularity_threshold,
                              high_simularity_threshold=high_simularity_threshold,
                              forget_usage_threshold=forget_usage_threshold,
                              forget_age_threshold=forget_age_threshold,
                              max_memory_size=max_memory_size)

            return ms

        else:

            return None
