import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from worlds.base_world import World as BaseWorld
import core.tools as tools
import becca_tools_control_panel.control_panel as cp
import worlds.world_tools as wtools

class World(BaseWorld):
    """ The watch world provides a sequence of video frames to the BECCA agent
    There are no actions that the agent can take that affect the world. 

    This world uses the OpenCV library. Installation instructions are here:
    http://docs.opencv.org/doc/tutorials/introduction/linux_install/linux_install.html#linux-installation
    """
    # This package assumes that it is located directly under the BECCA package 
    def __init__(self, lifespan=None):
        super(World, self).__init__()
        if lifespan is not None:
            self.LIFESPAN = lifespan
        # Flag indicates whether the world is in testing mode
        #self.short_test = False
        self.TEST = False
        self.VISUALIZE_PERIOD =  1e0
        # Flag determines whether to plot all the features during display
        self.print_all_features = False
        #self.fov_horz_span = 12
        #self.fov_vert_span = 9
        #self.name = 'watch_world_9x12_c'
        self.fov_horz_span = 20
        self.fov_vert_span = 15
        self.name = 'watch_world_15x20'
        print "Entering", self.name
        # Generate a list of the filenames to be used
        self.video_filenames = []
        extensions = ['.mpg', '.mp4', '.flv', '.avi']
        if self.TEST:
            test_filename = 'test_long.avi'
            truth_filename = 'truth_long.txt'
            self.video_filenames = []
            self.video_filenames.append(os.path.join(
                    'becca_world_watch', 'test', test_filename))
            self.ground_truth_filename = os.path.join('becca_world_watch', 
                                                      'test', truth_filename)
        else:
            self.data_dir_name = os.path.join('becca_world_watch', 'data')
            self.video_filenames = tools.get_files_with_suffix(
                    self.data_dir_name, extensions)
        self.video_file_count = len(self.video_filenames)
        print self.video_file_count, 'video files loaded.'
        # Initialize the video data to be viewed
        self.initialize_video_file()
        self.intensity_image = np.zeros((300, 400))

        self.num_sensors = 2 * self.fov_horz_span * self.fov_vert_span
        self.num_actions = 0
        self.sensors = np.zeros(self.num_sensors)
        #self.initialize_control_panel()
        self.frame_counter = 20000
        self.frames_per_step = 3
        self.frames_per_sec = 30.
        if self.TEST:
            self.surprise_log_filename = os.path.join('becca_world_watch', 
                                                      'log', 'surprise.txt')
            self.surprise_log = open(self.surprise_log_filename, 'w')

    def initialize_video_file(self):
        """ Queue up one of the video files and get it ready for processing """
        if self.video_file_count == 0:
            print 'Add some video files to the data directory'
        filename = self.video_filenames[
                np.random.randint(0, self.video_file_count)]
        print 'Loading', filename
        self.video_reader = cv2.VideoCapture(filename)
        self.clip_frame = 0

    def step(self, action): 
        """ Advance the video one time step and read and process the frame """
        self.previous_image = self.intensity_image.copy()
        self.previous_sensors = self.sensors.copy()
        for _ in range(self.frames_per_step):
            ((success, image)) = self.video_reader.read() 
        # Check whether the end of the clip has been reached
        if not success:
            if self.TEST:
                # Terminate the test
                self.video_reader.release()
                self.surprise_log.close()
                print 'End of test reached'
                tools.report_roc(self.ground_truth_filename, 
                                 self.surprise_log_filename, self.name)
                sys.exit()
            else:
                self.initialize_video_file()
                ((success, image)) = self.video_reader.read() 
        self.timestep += 1
        self.clip_frame += self.frames_per_step
        image = image.astype('float') / 256.
        # Convert the color image to grayscale
        self.intensity_image = np.sum(image, axis=2) / 3.
        # Crop the image to get rid of sidebars
        (image_height, image_width) = self.intensity_image.shape
        self.intensity_image = self.intensity_image[
                image_height / 4: 3 * image_height / 4,
                image_width / 4 : 3 * image_width / 4]
        # Convert the grayscale to center-surround contrast pixels
        center_surround_pixels = wtools.center_surround(
                self.intensity_image, self.fov_horz_span, self.fov_vert_span)
        unsplit_sensors = center_surround_pixels.ravel()
        self.sensors = np.concatenate((np.maximum(unsplit_sensors, 0), 
                                       np.abs(np.minimum(unsplit_sensors, 0))))
        reward = 0
        return self.sensors, reward
        
    def set_agent_parameters(self, agent):
        """ Manually set some agent parameters, where required """
        agent.VISUALIZE_PERIOD = self.VISUALIZE_PERIOD
        if self.TEST:
            # Prevent the agent from adapting during testing
            agent.BACKUP_PERIOD = 10 ** 9
            for block in agent.blocks:
                block.ziptie.COACTIVITY_UPDATE_RATE = 0.
                block.ziptie.JOINING_THRESHOLD = 2.
                block.ziptie.AGGLOMERATION_ENERGY_RATE = 0.
                block.ziptie.NUCLEATION_ENERGY_RATE = 0.
                for cog in block.cogs:
                    cog.ziptie.COACTIVITY_UPDATE_RATE = 0.
                    cog.ziptie.JOINING_THRESHOLD = 2.
                    cog.ziptie.AGGLOMERATION_ENERGY_RATE = 0.
                    cog.ziptie.NUCLEATION_ENERGY_RATE = 0.
                    cog.daisychain.CHAIN_UPDATE_RATE = 0.
        else:
            pass
        return
    
    def visualize(self, agent):
        """ Update the display to the user of the world's internal state """
        if self.TEST:
            # Save the surprise value
            surprise_val = agent.surprise_history[-1]
            time_in_seconds = str(float(self.clip_frame) / self.frames_per_sec)
            file_line = ' '.join([str(surprise_val), str(time_in_seconds)])
            self.surprise_log.write(file_line)
            self.surprise_log.write('\n')

        if (self.timestep % self.VISUALIZE_PERIOD != 0):
            return 
        print ' '.join([self.name, 'is', str(self.timestep), 'timesteps old.'])
        
        (projections, feature_activities) = agent.get_index_projections()

        # Make a composite plot showing the current state of the world
        max_blocks = 4
        max_cols = 6
        fig = plt.figure(16, facecolor=tools.LIGHT_COPPER, 
                         figsize=(16., 9.))
        plt.clf()
        sub1 = plt.subplot2grid((max_blocks, max_cols), 
                                (0,0), colspan=3, rowspan=3)
        plt.gray()
        plt.imshow(np.flipud(self.previous_image), interpolation='bicubic')
        (n_rows, n_cols) = self.previous_image.shape
        top = float(n_rows) / float(self.fov_vert_span + 2)
        bottom = (float(n_rows) * float(self.fov_vert_span + 1) / 
                  float(self.fov_vert_span + 2))
        left = float(n_cols) / float(self.fov_horz_span + 2)
        right = (float(n_cols) * float(self.fov_horz_span + 1) / 
                 float(self.fov_horz_span + 2))
        plt.plot(np.array([left, right, right, left, left]), 
                 np.array([top, top, bottom, bottom, top]), 
                 color=tools.LIGHT_COPPER, linewidth=4.)
        plt.ylim((0, n_rows))
        plt.xlim((0, n_cols))
        ax = plt.gca()
        ax.set_autoscale_on(False)
        sub1.xaxis.set_visible(False)
        sub1.yaxis.set_visible(False)

        # Update status window 
        sub_text = plt.subplot2grid((max_blocks, max_cols), 
                                    (max_blocks - 1, 1), colspan=2)
        sub_text.axis((0., 1., 0., 1.))
        sub_text.get_xaxis().set_visible(False)
        sub_text.get_yaxis().set_visible(False)

        time_string = wtools.duration_string(
                self.clip_frame / self.frames_per_sec)
        clip_time_text = ' '.join(('Clip time:', time_string))
        time_string = wtools.duration_string(
                self.timestep * self.frames_per_step / self.frames_per_sec)
        wake_time_text = ' '.join(('Wake time:', time_string))
        time_string = wtools.duration_string(
                agent.timestep * self.frames_per_step / self.frames_per_sec)
        life_time_text = ' '.join(( 'Life time:', time_string))

        sub_text.text(.1, .8, clip_time_text, 
                    color=tools.COPPER_SHADOW, size=14, ha='left', va='center')
        sub_text.text(.1, .6, wake_time_text, 
                    color=tools.COPPER_SHADOW, size=14, ha='left', va='center')
        sub_text.text(.1, .4, life_time_text, 
                    color=tools.COPPER_SHADOW, size=14, ha='left', va='center')

        # Display sensed image
        sub2 = plt.subplot2grid((max_blocks, max_cols), 
                                (max_blocks - 1, 3), colspan=1)
        plt.gray()
        sensed_image_array = wtools.visualize_pixel_array_feature(
                self.previous_sensors[:,np.newaxis], 
                fov_horz_span=self.fov_horz_span,
                fov_vert_span=self.fov_vert_span, array_only=True) 
        plt.imshow(sensed_image_array[0], interpolation='bicubic', 
                   vmin=.2, vmax=.8)
        sub2.xaxis.set_visible(False)
        sub2.yaxis.set_visible(False)
        
        
        # Make a copy of projections for finding the interpretation
        interpretation_by_feature = list(projections)
        for block_index in range(np.minimum(len(interpretation_by_feature),
                                            max_blocks - 1)):
            # Display each block's interpretation of the image
            sub_n = plt.subplot2grid((max_blocks, max_cols), 
                                     (max_blocks - 2 - block_index, 3),
                                     colspan=1)
            #plt.gray()
            interpretation = np.zeros((self.num_sensors, 1))
            for feature_index in range(len(interpretation_by_feature
                                           [block_index])):
                # This interpretation only shows the last time step in
                # the sequence
                #this_feature_interpretation = (
                #        interpretation_by_feature[block_index] 
                #        [feature_index][:self.num_sensors,-1][:,np.newaxis])
                # This interpretation shows all the time steps at once
                this_feature_interpretation = (
                        np.sum(interpretation_by_feature[block_index] 
                        [feature_index][:self.num_sensors,:], 
                        axis=1)[:,np.newaxis])
                interpretation = np.maximum(interpretation, 
                        this_feature_interpretation *
                        feature_activities[block_index][feature_index])

            interpreted_image_array = wtools.visualize_pixel_array_feature(
                    interpretation[:self.num_sensors],  
                    fov_horz_span=self.fov_horz_span,
                    fov_vert_span=self.fov_vert_span, array_only=True) 
            plt.imshow(interpreted_image_array[0], interpolation='bicubic', 
                       vmin=.2, vmax=.8)
            sub_n.xaxis.set_visible(False)
            sub_n.yaxis.set_visible(False)

        for block_index in range(np.minimum(len(agent.blocks), max_blocks)):
            # Display each block's feature activities
            sub_m = plt.subplot2grid((max_blocks, max_cols), 
                                     (max_blocks - 1 - block_index, 4),
                                     colspan=2)
            #plt.gray()
            block = agent.blocks[block_index]
            cable_activities = block.cable_activities
            num_cogs_in_block = len(block.cogs)
            #activity_array = np.reshape(cable_activities, 
            #                            (num_cogs_in_block,
            #                             block.max_bundles_per_cog)).T
            activity_array = np.reshape(cable_activities, 
                                        (num_cogs_in_block / 2, -1)).T
            plt.imshow(activity_array, aspect='auto', 
                           interpolation='nearest', vmin=0., vmax=.2,
                           cmap='copper')
            sub_m.xaxis.set_visible(False)
            sub_m.yaxis.set_visible(False)

        fig.show()
        fig.canvas.draw()
        plt.tight_layout()
        # Save the control panel image
        filename =  self.name + '_' + str(self.frame_counter) + '.png'
        full_filename = os.path.join('becca_world_watch', 'frames', 
                                     filename)
        self.frame_counter += 1
        plt.figure(fig.number)
        dpi = 80 # for a resolution of 720 lines
        #dpi = 120 # for a resolution of 1080 lines
        plt.savefig(full_filename, format='png', dpi=dpi, 
                    facecolor=fig.get_facecolor(), edgecolor='none') 

        if self.print_all_features:
            log_directory = os.path.join('becca_world_watch', 'log')
            wtools.print_pixel_array_features(projections, self.num_sensors,
                                              self.num_actions, 
                                              self.fov_horz_span,
                                              self.fov_vert_span, 
                                              directory=log_directory,
                                              world_name=self.name,
                                              interp='bicubic')
        return
