import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.io as ski
import sys

from worlds.base_world import World as BaseWorld
import core.tools as tools
import worlds.ffmpeg_tools as ft
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
        # Flag determines whether to plot all the features during display
        self.print_all_features = False
        self.fov_horz_span = 20
        self.fov_vert_span = 15
        self.name = 'watch_world_15x20'
        print "Entering", self.name
        # Generate a list of the filenames to be used
        self.frames_dir_name = os.path.join('becca_world_watch', 'frames')
        self.video_filenames = []
        extensions = ['.mpg', '.mp4', '.flv', '.avi', '.m4v']
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
        self.frame_counter = 20000
        self.frames_per_step = 3
        self.frames_per_sec = 30.

    def initialize_video_file(self):
        """ Queue up one of the video files and get it ready for processing """
        if self.video_file_count == 0:
            print 'Add some video files to the data directory'
        # Empty out the frames directory
        for filename in os.listdir(self.frames_dir_name): 
            if '.jpg' in filename:
                os.remove(os.path.join(self.frames_dir_name, filename))
        # Pick and load a new video file    
        filename = self.video_filenames[
                np.random.randint(0, self.video_file_count)]
        print 'Loading', filename
        # Break video into frames
        ft.break_movie(filename, self.data_dir_name, self.frames_dir_name)

        # Make a list of still image filenames
        self.frame_list = []
        for filename in os.listdir(self.frames_dir_name):
            if '.jpg' in filename:
                self.frame_list.append(filename)

        self.clip_frame = 0

    def step(self, action): 
        """ Advance the video one time step and read and process the frame """
        self.previous_image = self.intensity_image.copy()
        self.previous_sensors = self.sensors.copy()
        try:
            for _ in range(self.frames_per_step):
                frame_name = self.frame_list.pop(0)
            full_frame_name = os.path.join(self.frames_dir_name, frame_name)
            image = ski.imread(full_frame_name) 
        except:
            self.initialize_video_file()
            frame_name = self.frame_list.pop(0)
            full_frame_name = os.path.join(self.frames_dir_name, frame_name)
            image = ski.imread(full_frame_name) 
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
        # Boost the magnitude of the sensors
        self.sensors *= 10.
        self.sensors = np.minimum(1., self.sensors)
        reward = 0
        return self.sensors, reward
        
    def visualize(self, agent):
        """ 
        Update the display to the user of the world's internal state
        """

        # debug: this is the most convenient place to tweak the agent
        agent.display_interval = 1e1
        self.print_all_features = True
        
        if self.timestep % agent.display_interval != 0:
            return 
        print ' '.join([self.name, 'is', str(self.timestep), 'timesteps old.'])
        (projections, feature_activities) = agent.get_index_projections()

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
