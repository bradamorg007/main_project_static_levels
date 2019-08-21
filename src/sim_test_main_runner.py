import os
import pickle as pkl
from block import Block
import matplotlib.pyplot as plt

import STM_tests
import GA_vanilla_tests
import random_vanilla_tests
import pickle
import numpy as np
from visual_system import VisualSystem
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
PURPLE = (255, 0, 255)
SCREEN_WIDTH = 1300
SCREEN_HEIGHT = 600

def build_simulation():
    level7 = build_blueprint([50, 390, 490, 70, BLACK], multipliers=10, xPos_interval=50)
    level8 = build_blueprint([50, 350, 480, 70, BLACK], multipliers=10, xPos_interval=50)
    level2 = build_blueprint([50, 480, 580, 70, BLACK], multipliers=10, xPos_interval=50)
    level3 = build_blueprint([50, 450, 550, 70, BLACK], multipliers=10, xPos_interval=50)

    level4 = build_blueprint([900, 100, 300, 70, BLACK], multipliers=3, xPos_interval=50)
    level4_1 = build_blueprint([50, 450, 550, 70, BLACK], multipliers=5, xPos_interval=50)
    level4_12 = build_blueprint([650, 250, 530, 70, BLACK], multipliers=1, xPos_interval=50)
    level4_13 = build_blueprint([780, 130, 470, 70, BLACK], multipliers=1, xPos_interval=50)
    level4 = merge(level4_1, level4_12, level4_13, level4)

    level9 = build_blueprint([900, 150, 350, 70, BLACK], multipliers=3, xPos_interval=50)
    level9_1 = build_blueprint([50, 450, 550, 70, BLACK], multipliers=5, xPos_interval=50)
    level9_12 = build_blueprint([650, 250, 530, 70, BLACK], multipliers=1, xPos_interval=50)
    level9_13 = build_blueprint([780, 130, 470, 70, BLACK], multipliers=1, xPos_interval=50)
    level9 = merge(level9_1, level9_12, level9_13, level9)

    level1 = build_blueprint([900, 100, 300, 70, BLACK], multipliers=2, xPos_interval=50)
    level1_0 = build_blueprint([1150, 200, 400, 70, BLACK], multipliers=1, xPos_interval=50)
    level1_0_0 = build_blueprint([1290, 300, 500, 70, BLACK], multipliers=2, xPos_interval=50)
    level1_1 = build_blueprint([50, 450, 550, 70, BLACK], multipliers=5, xPos_interval=50)
    level1_12 = build_blueprint([650, 250, 530, 70, BLACK], multipliers=1, xPos_interval=50)
    level1_13 = build_blueprint([780, 130, 470, 70, BLACK], multipliers=1, xPos_interval=50)
    level1 = merge(level1_1, level1_12, level1_13, level1, level1_0, level1_0_0)

    level0 = build_blueprint([900, 300, 500, 70, BLACK], multipliers=2, xPos_interval=50)
    level0_0 = build_blueprint([1150, 400, 550, 70, BLACK], multipliers=1, xPos_interval=50)
    level0_0_0 = build_blueprint([300, 300, 530, 70, BLACK], multipliers=2, xPos_interval=50)
    level0_0_1 = build_blueprint([530, 450, 530, 70, BLACK], multipliers=1, xPos_interval=50)
    level0_1 = build_blueprint([50, 450, 550, 70, BLACK], multipliers=2, xPos_interval=50)
    level0_12 = build_blueprint([650, 250, 530, 70, BLACK], multipliers=1, xPos_interval=50)
    level0_13 = build_blueprint([780, 130, 470, 70, BLACK], multipliers=1, xPos_interval=50)
    level0 = merge(level0_1, level0_0_0, level0_0_1, level0_12, level0_13, level0, level0_0)

    level6_0 = build_blueprint([1150, 400, 500, 70, BLACK], multipliers=1, xPos_interval=50)
  #  level6_0_0 = build_blueprint([1260, 450, 550, 70, BLACK], multipliers=1, xPos_interval=50)
    level6_1 = build_blueprint([50, 450, 550, 70, BLACK], multipliers=5, xPos_interval=50)
    level6_12 = build_blueprint([650, 370, 530, 70, BLACK], multipliers=1, xPos_interval=50)
    level6_13 = build_blueprint([780, 300, 500, 70, BLACK], multipliers=1, xPos_interval=50)
    level6 = build_blueprint([900, 300, 500, 70, BLACK], multipliers=2, xPos_interval=50)
    level6 = merge(level6_1, level6_12, level6_13, level6, level6_0,)

    level5_1 = build_blueprint([50, 450, 550, 70, BLACK], multipliers=5, xPos_interval=50)
    level5_12 = build_blueprint([650, 370, 530, 70, BLACK], multipliers=1, xPos_interval=50)
    level5_13 = build_blueprint([780, 300, 500, 70, BLACK], multipliers=1, xPos_interval=50)
    level5 = build_blueprint([900, 300, 500, 70, BLACK], multipliers=3, xPos_interval=50)
    level5 = merge(level5_1, level5_12, level5_13, level5)

    levels = Block.generate(SCREEN_WIDTH, SCREEN_HEIGHT, 100, False, level2, level0, level5, level9, level4, level6, level0)
    """level5,
                                                                     level6, level7, level8, level9)"""

    return levels

def merge(*args):

    output = []

    for item in args:

        #item[-1][4] = (0,0,0)
        output = output + item

    return output

def build_blueprint(blueprint, multipliers, xPos_interval=None):

        level_array = []

        new_Xpos = 0
        pos = blueprint[0]
        for i in range(multipliers):


            if i > 0:
                new_blueprint = blueprint.copy()
                new_blueprint[0] = pos
                new_Xpos += blueprint[3] + xPos_interval
                new_blueprint[0] += new_Xpos

                if blueprint[0] < SCREEN_WIDTH:
                    level_array.append(new_blueprint)

            else:
                level_array.append(blueprint)

        #level_array[len(level_array)-1][4] = GREEN

        return level_array


def save_agent(agent):
    # save best agent
    agent.computeFitness()
    folder = "sim_data"
    filename = os.path.join(folder, 'best_agent')
    if os.path.isdir(folder) == False:
        os.mkdir(folder)

    pkl.dump(agent.functional_system, open(os.path.join(filename), 'wb'))
    print('NOTIFICATION: Best Agent has been saved to %s' % filename)

#=======================================================================================================================

filename = 'full_test_simulation1'
if os.path.isdir('labs') == False:
    os.mkdir('labs')

if os.path.isdir(os.path.join('labs',filename)) == False:
    os.mkdir(os.path.join('labs',filename))

""" Main Program """
""" Model parameters """
MODE = 'test'
EXTRACT_DATA = 'gen_test'
SIM_SPEED = 500
DRAW = False
save_first_winner = False
level_to_save = 0
generation_limit = 8
end_on_lap = 1
capture_filename = 'sim_test_main_capture'
attempts = 3
RUN_STM_ONLY = False

levels = build_simulation()
#levels = [levels[9], levels[0], levels[8], levels[1], levels[4], levels[3], levels[5], levels[6], levels[7]]

visual_system = VisualSystem.init(img_shape=(40, 40, 1),
                                  latent_dims=3,
                                  RE_delta=0.0,
                                  model_folder='CNND_main',
                                  start=1,
                                  MODE=MODE,
                                  preview_output=False
                                  )


tests = 100
gen_rec1 = np.zeros(len(levels))
gen_rec2 = np.zeros(len(levels))
gen_rec3 = np.zeros(len(levels))

for i in range(tests):

    STM = np.array(
                        STM_tests.run_simulation(DRAW=True, SCREEN_HEIGHT=SCREEN_HEIGHT, SCREEN_WIDTH=SCREEN_WIDTH,
                                       MODE=MODE, EXTRACT_DATA=EXTRACT_DATA, SIM_SPEED=1,
                                       save_first_winner=save_first_winner, level_to_save=level_to_save,
                                       generation_limit=generation_limit, end_on_lap=end_on_lap,
                                       capture_filename=capture_filename, levels=levels, attempts=1,
                                       save_trained_agent='main_agent', load_agent_filepath='sim_data/main_agent',
                                       high_simularity_threshold = 0.05,
                                       low_simularity_threshold =  0.7,
                                       min_lts=0.0, max_lts=0.21605951342292382,
                                       visual_system=visual_system)
                        )

    if RUN_STM_ONLY == False:
        GA_V = np.array(
                            GA_vanilla_tests.run_simulation(DRAW=DRAW, SCREEN_HEIGHT=SCREEN_HEIGHT, SCREEN_WIDTH=SCREEN_WIDTH,
                                           MODE='train', EXTRACT_DATA=EXTRACT_DATA, SIM_SPEED=SIM_SPEED,
                                           save_first_winner=save_first_winner, level_to_save=level_to_save,
                                           generation_limit=generation_limit, end_on_lap=end_on_lap,
                                           levels=levels, attempts=attempts)
                            )

        RAND_V = np.array(
                            random_vanilla_tests.run_simulation(DRAW=DRAW, SCREEN_HEIGHT=SCREEN_HEIGHT, SCREEN_WIDTH=SCREEN_WIDTH,
                                           MODE='train', EXTRACT_DATA=EXTRACT_DATA, SIM_SPEED=SIM_SPEED,
                                           save_first_winner=save_first_winner, level_to_save=level_to_save,
                                           generation_limit=generation_limit, end_on_lap=end_on_lap,
                                           levels=levels, attempts=attempts)
                            )
        gen_rec2 = np.add(gen_rec2, GA_V)
        gen_rec3 = np.add(gen_rec3, RAND_V)

    gen_rec1 = np.add(gen_rec1, STM)


    print('=======================================================')
    print()
    print('                     TEST %s of %s COMPLETE' % (i, tests))
    print()
    print('=======================================================')


data_package = {'STM': gen_rec1, 'GA_vanilla': gen_rec2, 'random_vanilla':gen_rec3}

pickle.dump(data_package, open( os.path.join('labs', filename, 'data_package'), "wb" ) )

gen_rec1 = np.divide(gen_rec1, tests)
gen_rec2 = np.divide(gen_rec2, tests)
gen_rec3 = np.divide(gen_rec3, tests)

fig = plt.figure(1)
plt.plot(gen_rec1, color='k', marker='o', markersize=5, linewidth=2)
plt.plot(gen_rec2, linestyle='--', marker='o', markersize=5, linewidth=2)
plt.plot(gen_rec3, linestyle='--', marker='o', markersize=5, linewidth=2)

plt.legend(['STM', 'Vanilla GA', 'Random_vanilla'])
plt.xlabel('Test Environments')
plt.ylabel('Number of Generations')
plt.grid()
plt.show()

fig.savefig(os.path.join('labs', filename, 'figure1'))




# level1 = build_blueprint([100, 450, 550, 100, BLACK], multipliers=4, xPos_interval=50)
# level2 = build_blueprint([400, 480, 580, 100, BLACK], multipliers=4, xPos_interval=50)
# level3 = build_blueprint([400, 350, 450, 100, BLACK], multipliers=4, xPos_interval=50)
# level4 = build_blueprint([400, 330, 430, 100, BLACK], multipliers=4, xPos_interval=50)
# level5 = build_blueprint([400, 250, 350, 100, BLACK], multipliers=7, xPos_interval=50)
# l61 = build_blueprint([0, 50, 550, 100, BLACK], multipliers=3, xPos_interval=50)
# level6 = merge(l61, level2)
#
# levels = Block.generate(SCREEN_WIDTH, SCREEN_HEIGHT, 100, False, level1)




