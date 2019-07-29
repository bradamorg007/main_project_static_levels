from old_code.GA import GA

class NeuroEvolution:

    def __init__(self):

        self.generation_count = 0


    def run(self, level_manager, agents, active_blocks, visual_system=None, memory_system=None):


        if level_manager.mode == 'train' or level_manager.mode == 'test':
            # determine closest block
            global_xPos = agents.not_sprites[0].rect.x
            closest_block = self.get_closest_block(active_blocks, global_xPos)

            # agents think using nn then update their positions
            # check if agents a block. if so remove from active_agents and add to dead agents
            i = 0
            while i < len(agents.not_sprites):

                agent = agents.not_sprites[i]


                if level_manager.mode == 'test' and visual_system is not None:
                    # ignore blackout screen at beginning

                    is_familiar = visual_system.is_familiar()

                    if is_familiar == False and memory_system is not None:
                        # get latent representation from vs
                        latent_representation, _ = visual_system.generate_latent_representation()
                        memory, action = memory_system.query(latent_representation)

                        if memory is not None and action == 'memory_to_fs_system_switch':

                            a = 0
                        elif memory is None and action == 'adaption_using_medium_memory_as_init_foundation':
                            pass

                        elif memory is None and action == 'adaption_using_low_memory_and_random_init_foundation':
                            pass



                    print('visual system is frame familiar: %s' % is_familiar)



                agent.think(closest_block, level_manager.SCREEN_WIDTH, level_manager.SCREEN_HEIGHT)
                agent.update(closest_block, level_manager.SCREEN_HEIGHT)

                if closest_block.hit(agent) or agent.off_screen(level_manager.SCREEN_HEIGHT,
                                                                level_manager.SCREEN_WIDTH):
                    agent.computeFitness()
                    agents.splice(i)
                    i -= 1

                i += 1

            # check if all active agents are dead, the perform GA and reset game level and epochs
            if len(agents.not_sprites) == 0:

                if level_manager.mode == 'test':
                    dead_agent = agents.dead_agents[0]
                    dead_agent.reset(screen_height=level_manager.SCREEN_HEIGHT)
                    new_population = [dead_agent]
                    agents.update_arrays(input=new_population)

                else:
                    new_population = GA.produceNextGeneration(population=agents.dead_agents,
                                                              screen_width=level_manager.SCREEN_WIDTH,
                                                              screen_height=level_manager.SCREEN_HEIGHT)

                    agents.update_arrays(new_population)

                self.generation_count += 1
                active_blocks = level_manager.level_reset(active_blocks)
                level_manager.RESET_FLAG = True
                print('generation = %s population size = %s epoch = %s / %s' % (
                self.generation_count, len(new_population), level_manager.epoch_count, level_manager.epochs))


        return level_manager, agents, active_blocks


    def get_closest_block(self, active_blocks, global_xPos):
        closest_block = None

        for i in range(len(active_blocks)):
            if global_xPos < active_blocks[i].bottom_block.rect.right:
                closest_block = active_blocks[i]
                active_blocks[i].top_block.image.fill((255, 0, 0))
                break

        return closest_block