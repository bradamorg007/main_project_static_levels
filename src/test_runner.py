import simulation_test
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from visual_system import VisualSystem



def progressBar(value, endvalue, bar_length=20):
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()


if __name__ == '__main__':

    filename = 'GEN'
    if os.path.isdir('labs') == False:
        os.mkdir('labs')

    if os.path.isdir(os.path.join('labs','test_runner_results')) == False:
        os.mkdir(os.path.join('labs','test_runner_results'))

    visual_system = VisualSystem.init(img_shape=(40, 40, 1),
                                      latent_dims=3,
                                      RE_delta=0.0,
                                      model_folder='CNND_sim_test',
                                      start=1,
                                      MODE='test',
                                      preview_output=False
                                     )

    tests = 60
    target_generation = 10

    memory_init_fitness_vals = []
    rand_init_fitness_vals = []
    memory_init_generation_count = []
    rand_init_generation_count = []
    count = 0
    i = 0
    avg_fitness = None
    generation_count = None
    while i < tests:

        if i % 2 == 0:
            rand = 0.0
            med_memory = 1.0
            avg_fitness, generation_count = simulation_test.run_model(MODE='test', data_extract=True, SIM_SPEED=1, generation_limit=10,
                                                    run_time=600, target_levels=[1], print_data=True,
                                                    visual_system=visual_system,
                                                    target_generation=target_generation,
                                                    med_sim_init_rand_agent_percenatge=rand,
                                                    med_sim_init_med_memory_agent_percenatge=med_memory,
                                                    med_sim_init_med_mutation_rate=0.1,
                                                    med_sim_init_rand_mutation_rate=0.0)

            visual_system.clean()
            print('rand agent percentage: %s medium memory percentage: %s' % (rand, med_memory))

            if avg_fitness == 'ERROR_AGENT_DEATH'or avg_fitness is None:
                tests -= 2

            else:
                memory_init_fitness_vals.append(avg_fitness)
                memory_init_generation_count.append(generation_count)



        elif i % 2 == 1:
            rand = 1.0
            med_memory = 0.0
            avg_fitness, generation_count = simulation_test.run_model(MODE='test', data_extract=True, SIM_SPEED=1, generation_limit=10,
                                                    run_time=600, target_levels=[1], print_data=True,
                                                    target_generation=target_generation,
                                                    visual_system=visual_system,
                                                    med_sim_init_rand_agent_percenatge=rand,
                                                    med_sim_init_med_memory_agent_percenatge=med_memory,
                                                    med_sim_init_med_mutation_rate=0.0,
                                                    med_sim_init_rand_mutation_rate=0.0
                                                    )

            visual_system.clean()
            print('rand agent percentage: %s medium memory percentage: %s' % (rand, med_memory))
            if avg_fitness == 'ERROR_AGENT_DEATH' or avg_fitness is None:
                tests -= 2
            else:
                rand_init_fitness_vals.append(avg_fitness)
                rand_init_generation_count.append(generation_count)

        if avg_fitness != 'ERROR_AGENT_DEATH' or avg_fitness is not None:
            print('test = %s / %s' % (i, tests))
            count += 1
            i += 1
        else:
            print('ERROR TEST RUNNER: Agent death or ERROR has occurred system is restarting tests')

    max = 0
    min = 0

    max_mem = np.max(memory_init_fitness_vals)
    min_mem = np.min(memory_init_fitness_vals)

    max_rand = np.max(rand_init_fitness_vals)
    min_rand = np.min(rand_init_fitness_vals)

    if max_mem > max_rand:
        max = max_mem
    else:
        max = max_rand

    if min_mem < min_rand:
        min = min_mem
    else:
        min = min_rand


    memory_init_final_vals = []
    rand_init_final_vals = []

    for item in memory_init_fitness_vals:
        x = (item - min) / (max - min)
        memory_init_final_vals.append(x)

    for item in rand_init_fitness_vals:
        x = (item - min) / (max - min)
        rand_init_final_vals.append(x)


    fig = plt.figure(1)
    plt.plot(memory_init_final_vals, color='k', marker='o')
    plt.plot(rand_init_final_vals, color='r', marker='o')
    plt.title('Average fitness of Agents using random and memory assisted adaptation')
    plt.xlabel('Number of Tests')
    plt.ylabel('Average fitness')
    plt.legend(['Memory Assisted adaptation','Random Assisted adaptation' ])
    plt.grid()

    fig2 = plt.figure(2)
    plt.plot(memory_init_generation_count, color='k', marker='o')
    plt.plot(rand_init_generation_count, color='r', marker='o')
    plt.title('Number of Generations for Agents to complete a Level using random and memory assisted adaptation')
    plt.xlabel('Number of Tests')
    plt.ylabel('Number of Generations')
    plt.legend(['Memory Assisted adaptation','Random Assisted adaptation' ])
    plt.grid()

    plt.show()

    fig.savefig(os.path.join('labs','test_runner_results', filename + 'avg_fitness'))
    fig2.savefig(os.path.join('labs', 'test_runner_results', filename + 'gen_count'))





