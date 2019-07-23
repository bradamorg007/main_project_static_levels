from block import Block
from visual_system import VisualSystem
from memory_system import MemorySystem
import numpy as np
import json
import os


class Colors:

    def __init__(self):
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED =   (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE =  (0, 0, 255)


class LevelManager:

    def __init__(self, FPS, game_len, epochs, number_of_blocks,
                 buffer_size, blueprints, override_gap_size,
                 override_block_width, override_block_speed,
                 screen_dimensions=(300, 300), batch_reset='seed',
                 optional_script_build_args="percentage", mode='train',
                 load_agent_filepath = None,
                 capture_mode_override=False,
                 data_augmentation=False,
                 y_top_jitter_probability=None, y_bottom_jitter_probability=None, width_jitter_probability=None,
                 y_top_jitter_amount=1.0, y_bottom_jitter_amount=1.0, width_jitter_amount=1.0):

        if mode != 'train' and mode != 'capture' and mode != 'test':
            raise ValueError('ERROR LevelManager __init__: Illegal Argument Exception the mode must be either train or capture')


        self.mode = mode
        self.load_agent_filepath = load_agent_filepath
        self.capture_mode_override =capture_mode_override
        self.colors = Colors()
        self.FPS = FPS
        self.game_len = game_len
        self.epochs = epochs
        self.epoch_count = 0

        self.data_augmentation = data_augmentation
        self.y_top_jitter = y_top_jitter_probability
        self.y_bottom_jitter = y_bottom_jitter_probability
        self.width_jitter = width_jitter_probability
        self.y_top_jitter_amount = y_top_jitter_amount
        self.y_bottom_jitter_amount = y_bottom_jitter_amount
        self.width_jitter_amount = width_jitter_amount

        # Override fields will be used if a random type blueprint is detected or if user wants all blueprints made randomly
        self.block_width = override_block_width
        self.block_speed = override_block_speed
        self.override_gap_size = override_gap_size
        self.override_block_width = override_block_width
        self.override_block_speed = override_block_speed

        self.buffer_size = buffer_size
        self.buffer = []
        self.batch_reset = batch_reset
        self.level_script = []
        self.level_script_bin = []
        self.folder_bin = 'AE_data'

        self.number_of_blocks = number_of_blocks
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = screen_dimensions
        self.block_display_freq_per_epoch = np.floor((self.FPS * self.game_len) / self.number_of_blocks)
        self.END_SIMULATION_FLAG = False
        self.RESET_FLAG = False
        self.TERMINATE_FLAG = False

        pattern = ['ordered', "percentage",'percentage_ordered' ]

        match = False
        for p in pattern:
            if p == optional_script_build_args:
                match = True
                break

        if match == False:
            self.optional_script_build_args = "percentage"
        else:
            self.optional_script_build_args = optional_script_build_args

        sum = 0
        if isinstance(blueprints, list):
            sum = self.blueprint_validation(blueprints)
            self.blueprints = blueprints
        elif blueprints == 'random':
            self.blueprints = blueprints
        else:
            raise ValueError('ERROR Level_manager __init__: value of argument blueprints must be set to random or contain a list of valid blueprints')

        self.sum = sum

        # build level script and load in first batch to buffer
        self.build_level_script(sum)
        if self.data_augmentation:
            self.apply_basic_data_augmentation()
        self.buffer_to_script_request()

    def compute_max_frames(self, capture_first_epoch_only):

        if capture_first_epoch_only:
            self.max_frames_one_epoch = np.floor(self.FPS * self.game_len)
            return self.max_frames_one_epoch

        else:
            self.max_frames_all_epochs = np.floor((self.FPS * self.game_len) * self.epochs)
            return self.max_frames_all_epochs


    def build_level_script(self, sum):
        # build the level script based on the blueprints

        # can take in a blue print to check if it needs random generation or not
        def make_blueprint_if_random(blueprint=None):

            if blueprint == None or blueprint[0] == 'r':
                topStart = int(np.round(np.random.rand() * (self.SCREEN_HEIGHT - self.override_gap_size)))
                min = topStart + self.override_gap_size
                bottomStart = int(np.floor(np.random.rand() * (self.SCREEN_HEIGHT - min + 1)) + min)

                # build the blue print from randomly generated information
                blueprint = [topStart, bottomStart, None, self.block_width, self.block_speed, None, 'random']

            return blueprint


        if self.blueprints == 'random':

            for i in range(self.number_of_blocks):
                self.level_script.append(make_blueprint_if_random())
        else:

            if self.optional_script_build_args == "ordered":
                for i in range(self.number_of_blocks):
                    for blueprint in self.blueprints:
                        self.level_script.append(make_blueprint_if_random(blueprint))

            elif self.optional_script_build_args == "percentage":
                for i in range(self.number_of_blocks):
                        self.level_script.append(make_blueprint_if_random(self.selector(self.blueprints, sum)))

            elif self.optional_script_build_args == 'percentage_ordered':
                for i in range(self.number_of_blocks):
                        self.level_script.append(make_blueprint_if_random(self.selector(self.blueprints, sum)))

                # sort based on percentage in descending order elemets with highest percentage first
                func = lambda x : x[5]
                self.level_script.sort(key=func, reverse=True)



    def apply_basic_data_augmentation(self):

        new_live_script = []
        for blueprint in self.level_script:

            new_blueprint = blueprint.copy()
            # add jitter here to provide some basic data augmentation
            execute_y_top_jitter = np.random.rand()
            execute_y_bottom_jitter = np.random.rand()
            execute_width_jitter = np.random.rand()

            if self.y_top_jitter != None and execute_y_top_jitter <= self.y_top_jitter:

                new_blueprint[0] = self.augment_data(input=blueprint[0],
                                                     max=self.y_top_jitter_amount,
                                                     padding=1,
                                                     check_type='height')

            if self.y_bottom_jitter != None and execute_y_bottom_jitter <= self.y_bottom_jitter:

                new_blueprint[1] = self.augment_data(input=blueprint[1],
                                                     max=self.y_bottom_jitter_amount,
                                                     padding=1,
                                                     check_type='height')

            if self.width_jitter != None and execute_width_jitter <= self.width_jitter:

                new_blueprint[3] = self.augment_data(input=blueprint[3],
                                                     max=self.width_jitter_amount,
                                                     padding=1,
                                                     check_type='width')


            new_live_script.append(new_blueprint)

        self.level_script = new_live_script


    def augment_data(self, input, max, padding, check_type):
        done = False
        while done == False:
            input = input + np.random.randint(low=-max, high=max)

            if check_type == 'height':
                if input > 0+padding and input < self.SCREEN_HEIGHT-padding:
                    done = True
            elif check_type == 'width':
                if input > 0+padding and input < self.SCREEN_WIDTH-padding:
                    done = True

        return input


    def draw_jitter_amount(self, max):
        done = False
        draw = 0
        while done == False:

            draw = np.random.normal(0, 1, 1)

            if draw <= max and draw >= -max:
                print(draw)
                done = True

        return draw


    def level_script_pull_request(self):
        blueprint = self.level_script.pop(0)

        if self.batch_reset == 'seed':
            self.level_script_bin.append(blueprint)

        return blueprint


    def buffer_to_script_request(self):
        # if buffer is empty it will fill it.
        # elif it will fill all available positions up to the buffer size
        # else if buffer is full it will pop first element off and new item to end like a que

        def append_buffer(loop_size):

            for i in range(loop_size):

                blueprint = self.level_script_pull_request()
                block_obj = blueprint_reader(blueprint)

                if block_obj == None:
                    raise ValueError(
                        'ERROR: buffer to script request: blueprint %s contains unreadable input signatures' % i)

                self.buffer.append(block_obj)


        def blueprint_reader(blueprint):

            y_top, y_bottom, max_gapSize, width, speed, _, class_label = blueprint

            # make the block generically
            if width == 'random' and speed == 'random':
                width = self.block_width
                speed = self.block_speed

            block = Block(SCREEN_WIDTH=self.SCREEN_WIDTH, SCREEN_HEIGHT=self.SCREEN_HEIGHT,
                          width=width, speed=speed,class_label=class_label)


            if isinstance(y_top, int) and isinstance(y_bottom, int):

                block.manual_config(topStart=y_top, bottomStart=y_bottom)
                block.build()

            else:
                return None

            return block


        if len(self.buffer) == 0:

            append_buffer(loop_size=self.buffer_size)

        elif len(self.buffer) < self.buffer_size:

            diff = self.buffer_size - len(self.buffer)
            append_buffer(loop_size=diff)

        elif len(self.buffer) == self.buffer_size:
            pass
           # append_buffer(loop_size=1)

        elif len(self.buffer) > self.buffer_size:

            raise ValueError("ERROR FATAL buffer_to_script_request: current buffer exceeds buffer size limit")

        else:
            raise ValueError("ERROR FATAL buffer_to_script_request: unexpected error detected please investigate")


    def buffer_pull_request(self):

        output_obj = self.buffer.pop(0)

        if len(self.level_script) > 0:
            self.buffer_to_script_request()

        return output_obj


    def selector(self, input, sum):

        index = 0
        r = np.round(np.random.rand() * sum)

        while r >= 0:
            r = r - input[index][len(input[index])-2]

            if r > 0:
                index = index + 1

        selection = input[index]

        if selection == None:
            raise ValueError("ERROR GA: Parent in select parent method is undefined this is due to the indexing")

        return selection


    def blueprint_validation(self, blueprints):

        sum = 0
        for blueprint in blueprints:

            if len(blueprint) != 7:
                raise ValueError('ERROR LevelManager: Not enough items in blueprint len = %s but should 7 ' % len(blueprint))

            if blueprint[0] != 'r':

                if blueprint[0] < 0 or blueprint[0] > blueprint[1]:
                    raise ValueError('ERROR LevelManager: Top block can not be smaller than 0 or be greater than the bottom blocks y position')

                if blueprint[1] > self.SCREEN_HEIGHT or blueprint[1] < blueprint[0]:
                    raise ValueError('ERROR LevelManager: Bottom block can not be greater than SCREEN_HEIGHT or be smaller than the top blocks y position')

                if blueprint[2] is not None:
                    raise ValueError('Error BluePrint sytax: predefined object positions do not require manually specified gap size')

            else:

                if blueprint[2] is None:
                    raise ValueError('Error BluePrint sytax position [2]: randomly defined object positions require a manually specified gap size')


            if blueprint[3] < 1:
                raise ValueError()

            if blueprint[5] < 0 or blueprint[5] > 100:
                raise ValueError()

            sum += blueprint[5]

        if sum != 100:
            raise ValueError()

        return sum


    def monitor(self, active_blocks, frame_count):

        # checks when to add new object to game from buffer
        # checks if obj is off screen and needs removing
        # chekcs if reached end of epoch, resest level_script and increments epoch counter




        if len(self.level_script) == 0 and len(self.buffer) == 0:

            if self.batch_reset == 'seed':
                self.level_script = self.level_script_bin
                self.level_script_bin = []
                self.buffer_to_script_request()


            else:
                self.build_level_script(self.sum)
                self.buffer_to_script_request()

            self.epoch_count += 1
            print('\n Epoch Completed: epochs = %s of %s' % (self.epoch_count, self.epochs))

        if self.epoch_count == self.epochs:
            self.END_SIMULATION_FLAG = True
            active_blocks = [active_blocks.pop(0)]

        if self.END_SIMULATION_FLAG == False and frame_count % self.block_display_freq_per_epoch == 0:
            #add new blocks
            active_blocks.append(self.buffer_pull_request())

        # remove offscreen blocks
        i = 0
        while i < len(active_blocks):
            if active_blocks[i].offscreen():
                active_blocks.pop(i)
                i -= 1
            i += 1

        if len(active_blocks) == 0:
            self.TERMINATE_FLAG = True


        return active_blocks


    def level_reset(self, active_blocks):

        self.epoch_count = 0

        if self.batch_reset == 'seed':
            # reset level from the start
            if len(self.level_script_bin) > 0:

                # clear the buffer for new batch. the add bin back to level_script
                self.buffer = []
                for i in range(len(self.level_script_bin)-1, -1, -1):
                    self.level_script.insert(0, self.level_script_bin[i])

                self.level_script_bin = []
                # refill buffer
                self.buffer_to_script_request()

            else:
                raise ValueError("ERROR LevelManager level reset: batch reset is True but there is nothing in the levelscript bin to reset with")

        else:
            # if not seed then init a new random init

            self.build_level_script(self.sum)
            self.buffer_to_script_request()

        return []


    def save_config(self, save_folder_path):

        if not os.path.isdir(os.path.join(self.folder_bin, save_folder_path)):
            os.mkdir(os.path.join(self.folder_bin, save_folder_path))

        dict = self.__dict__
        dict = dict.copy()
        dict.pop('colors')
        dict.pop('buffer')
        dict.pop('level_script')
        dict.pop('level_script_bin')

        filename = os.path.join(self.folder_bin, save_folder_path, 'level_manager_config.json')

        with open(filename, 'w') as f:
            json.dump(dict, f, indent=4, sort_keys=True)


    @staticmethod
    def load_config(filepath, mode):

        with open(filepath) as json_file:
            d = json.load(json_file)

        object = LevelManager(FPS=d.get('FPS'), game_len=d.get('game_len'), epochs=d.get('epochs'),
                              number_of_blocks=d.get('number_of_blocks'), buffer_size=d.get('buffer_size'),
                              blueprints=d.get('blueprints'), override_gap_size=d.get('override_gap_size'),
                              override_block_width=d.get('override_block_width'),
                              override_block_speed=d.get('override_block_speed'), screen_dimensions=(d.get('SCREEN_WIDTH'), d.get('SCREEN_HEIGHT')),
                              batch_reset=d.get('batch_reset'), optional_script_build_args=d.get('optional_script_build_args'),
                              mode=mode, capture_mode_override=d.get('capture_mode_override'),
                              data_augmentation=d.get('data_augmentation'),
                              y_top_jitter_probability=d.get('y_top_jitter_probability'),
                              y_bottom_jitter_probability=d.get('y_bottom_jitter_probability'),
                              width_jitter_probability=d.get('width_jitter_probability'),
                              y_top_jitter_amount=d.get('y_top_jitter_amount'),
                              y_bottom_jitter_amount=d.get('y_bottom_jitter_amount'),
                              width_jitter_amount=d.get('width_jitter_amount'))

        return object


    def generate_visual_system(self, img_shape, latent_dims, RE_delta,
                     model_folder, start):

        if self.mode == 'test':
           vs =  VisualSystem(img_shape, latent_dims, RE_delta,
                         model_folder, start)

           return vs

        else:
            return None


    def generate_memory_system(self, low_simularity_threshold,high_simularity_threshold,
                               forget_usage_threshold,forget_age_threshold,
                               max_memory_size):

        if self.mode == 'test':
            ms = MemorySystem(low_simularity_threshold=low_simularity_threshold,
                              high_simularity_threshold=high_simularity_threshold,
                              forget_usage_threshold=forget_usage_threshold,
                              forget_age_threshold=forget_age_threshold,
                              max_memory_size=max_memory_size)

            return ms

        else:

            return None





if __name__ == "__main__":

    l = LevelManager.load_config('AE_data/data_0_static/level_manager_config.json', mode='train')



    blueprints = [[5, 20, None, 10, 60, 80, 'seen'],
                  [40, 55, None, 10, 1, 20, 'unseen'],
                  ]

    level_manager = LevelManager(FPS=60,
                                 game_len=20,
                                 epochs=10,
                                 number_of_blocks=50,
                                 buffer_size=10,
                                 blueprints=blueprints,

                                 override_gap_size=80,
                                 override_block_width=50,
                                 override_block_speed=4,

                                 batch_reset='no',
                                 optional_script_build_args='percentage'
                                 )

    block_obj = level_manager.buffer_pull_request()

    a = 0