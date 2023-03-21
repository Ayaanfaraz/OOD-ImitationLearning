#!/usr/bin/env python3

import carla
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pygame
import queue
import random
import sys
import time

from carla import ColorConverter as cc
from matplotlib import image
from tqdm import tqdm

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

IMAGE_HEIGHT = 300
IMAGE_WIDTH = 300

TIMESTEPS_PER_EPISODE = 400
NUM_EPISODES = 25

measurements_list = []
images_list = []

# Spawn points
x_list = [313, 354, 320, 352, 411, -92, -272.4, -328, -379, -18.9, -102.78, 12, 112.36, 209.98, 301.89, -354.77, 104.56, 160.7, -69.66, -486.9, -513.9, -425.54, -460.57]
y_list = [-263, -68, -118, -205, -222, -173, -87.8, -81, -21.6, 267.66, 377.93, -232.06, -360.25, -367.7, -356.8, 407.2, -376.74,-386.13, 383.58, 324.66, 216.74, 396.95, 12]

#=====================================
#   Sync Mode, Synchronize timesteps
#=====================================
class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world,weather, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None
        self.weather = weather
    def __enter__(self):
        self._settings = self.world.get_settings()

        weather = carla.WeatherParameters(
        cloudiness=0.0,
        precipitation=0.0,
        sun_altitude_angle=90.0)

        self.world.set_weather(weather)

        self.world.set_weather(self.weather)
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        try:
            data = [self._retrieve_data(q, timeout) for q in self._queues]
            assert all(x.frame == self.frame for x in data)
            return data
        except:
            time.sleep(10)
            self.frame = self.world.tick()
            data = [self._retrieve_data(q, timeout) for q in self._queues]
            assert all(x.frame == self.frame for x in data)
            return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


#=====================================
#   Spawn actors and build sensors
#=====================================
class CarEnv:
    num_timesteps = 0

    def __init__(self,town):
        self.client = carla.Client('127.0.0.1', 12000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(town)
        self.map = self.world.get_map()  
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]  
        self.tm = self.client.get_trafficmanager(8000)   
        self.tm_port = self.tm.get_port()


    def reset(self):
        self.collision_hist = []
        self.laneinvasion_hist = []
        self.obstacle_data=[]    
        self.actor_list = []
        self.num_timesteps = 1
        self.i = random.randint(0, len(x_list)-1)  ##picking random spawn points from the x_list and y_list
        
        try:
            self.waypoints = self.client.get_world().get_map().generate_waypoints(distance=3.0)
        except:
            time.sleep(10)

        self.waypoints = self.client.get_world().get_map().generate_waypoints(distance=3.0)

        # Spawn the vehicle at the specific locations, on the curve 
        self.spawn_point = carla.Transform(carla.Location(x=x_list[self.i], y=y_list[self.i], z=0.598),carla.Rotation(pitch=0.0, yaw=0.0, roll=0.000000))

        # self.spawn_point = random.choice(self.waypoints).transform #Used to be waypoint[0]
        self.spawn_point.location.z += 2
        self.vehicle = self.world.spawn_actor(self.model_3, self.spawn_point)  ## changed for adding waypoints

        self.actor_list.append(self.vehicle)
        # self.vehicle.set_autopilot(True)
        self.vehicle.set_autopilot(True, self.tm_port)
        self.tm.ignore_lights_percentage(self.vehicle,100)  ##ignore red lights

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        # self.rgb_cam.set_attribute('sensor_tick', str(tick_sensor))
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")  ## fov, field of view

        self.ss_cam = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        # self.ss_cam.set_attribute('sensor_tick', str(tick_sensor))
        self.ss_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.ss_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.ss_cam.set_attribute("fov", f"110")

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.rgb_sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.rgb_sensor)

        self.sem_sensor = self.world.spawn_actor(self.ss_cam, transform, attach_to=self.vehicle)
        self.cc_segm = carla.ColorConverter.CityScapesPalette
        self.actor_list.append(self.sem_sensor)

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        self.vehicle.set_autopilot(True)
        
        colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.colsensor.listen(lambda event: self.collision_data(event))
        self.actor_list.append(self.colsensor)
        
        laneinvsensor = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.laneinvsensor = self.world.spawn_actor(laneinvsensor, transform, attach_to=self.vehicle)
        self.laneinvsensor.listen(lambda event: self.laneinvasion_data(event))
        self.actor_list.append(self.laneinvsensor)

        obstacle_detector = self.world.get_blueprint_library().find('sensor.other.obstacle')
        obstacle_detector.set_attribute("distance", f"17")
        self.obstacle_detector = self.world.spawn_actor(obstacle_detector, transform, attach_to=self.vehicle)
        self.obstacle_detector.listen(lambda event: self.obstacle_hist(event))
        self.actor_list.append(self.obstacle_detector)
        
    def collision_data(self, event):
        self.collision_hist.append(event)

    def laneinvasion_data(self, event):
        self.laneinvasion_hist.append(event)

    def obstacle_hist(self, event):
        self.obstacle_data.append(event)

    def get_speed(self):
        velocity = self.get_velocity()
        return np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)    
    
    #=====================================
    #   Returns RGB Ground Truth
    #=====================================
    def draw_image_rgb(self, image):
        array_rgb = np.array(image.raw_data, dtype=np.dtype("uint8"))
        array_rgb = np.reshape(array_rgb, (image.height, image.width, 4))
        array_rgb = array_rgb[:, :, :3]
        return array_rgb

    #=====================================
    #   Returns Semantic Ground Truth
    #=====================================
    def draw_image_seg(self, image):
        image.convert(cc.CityScapesPalette)
        array_seg = np.array(image.raw_data, dtype=np.dtype("uint8"))
        array_seg = np.reshape(array_seg, (image.height, image.width, 4))
        array_seg = array_seg[:, :, :3]
        return array_seg
        

    def _on_invasion(self, event):
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ["%r" % str(x).split()[-1] for x in lane_types]
        self.hud.notification("Crossed line %s" % " and ".join(text))
        self.laneinvasion_hist.append(text) #ali


    #=====================================
    #  Saves images and measurement per timestep.
    #=====================================
    def step(self,sync_mode):

        # Get vehicle controls and sensors.
        control = self.vehicle.get_control()
        location = self.vehicle.get_transform().location
        rotation = self.vehicle.get_transform().rotation
        acceleration = self.vehicle.get_acceleration()
        velocity = self.vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  

        # Detect collision or lane invasion events in buffer.
        collision = 1 if len(self.collision_hist) != 0 else 0
        laneinvasion = 1 if len(self.laneinvasion_hist) != 0 else 0

        # Create list of controls to save as measurement labels.
        data = [float(control.steer),
                float(control.throttle),
                float(control.brake),
                float(location.x),
                float(location.y),
                float(rotation.pitch),
                float(rotation.yaw),
                float(rotation.roll),
                float(acceleration.x),
                float(acceleration.y),
                float(acceleration.z),
                float(speed),
                float(velocity.x),
                float(velocity.y),
                float(velocity.z),
                float(collision),
                float(laneinvasion)]

        done = False
        
        # Stop collecting data and end episode on collision.
        if collision:
            self.vehicle.set_autopilot(False)
            done = True

        # Stop collecting data if we go past total episode length.
        if self.num_timesteps >= TIMESTEPS_PER_EPISODE:
            self.vehicle.set_autopilot(False)
            done = True

        # Query an rgb and semantic image at a timestep if possible.
        try:
            _, image_rgb, image_seg = sync_mode.tick(timeout=20.0)
        except:
            print("error")
            return done, True

        image_rgbs = env.draw_image_rgb(image_rgb)
        semantic_segmentation = env.draw_image_seg(image_seg)
        measurements_list.append(data)
        images_list.append((image_rgbs, semantic_segmentation))
        
        cv2.imshow("Ground Truth RGB",image_rgbs)
        cv2.waitKey(1)
        
        cv2.imshow("Semantic Segmentation", semantic_segmentation)
        cv2.waitKey(1)
        return done, None

#=====================================
#   Episode based loop to save data. 
#=====================================
if __name__ == '__main__':
    
    collectionTowns = ['Town01','Town02','Town03','Town04']
    collectionWeather = [carla.WeatherParameters.ClearNoon, 
                        carla.WeatherParameters.HardRainSunset,
                        carla.WeatherParameters.CloudyNoon,
                        carla.WeatherParameters.ClearSunset]

    for town in collectionTowns:
        for weather in collectionWeather:
            env = CarEnv(town)

            for episode in tqdm(range(0, NUM_EPISODES), unit='episodes'):
                pygame.init()

                # Clear collision and obstacle distance buffers. 
                env.collision_hist = []
                env.obstacle_data = []

                step = 0

                # Reset environment and sensors.
                env.reset()
                done = False

                print("Data Collection Episode: %d", episode)

                # Synchronously iterate through sensors. (Collect data)
                with CarlaSyncMode(env.world, weather, env.rgb_sensor, env.sem_sensor, fps=10) as sync_mode:
                    
                    # Get ground truth synchronous RGB and Semantic image.
                    _, image_rgb, image_seg = sync_mode.tick(timeout=20.0)

                    while not done:
                        done, err = env.step(sync_mode)

                        # Increase timestep counters. 
                        step += 1
                        env.num_timesteps = step

                        # Error handling for server instability
                        # If something weird happens terminate episode.
                        if err is True:
                            print("ERROR!!!!")
                            break

                for actor in env.actor_list:
                    actor.destroy()
                pygame.quit()
            

            # Store images and measurements given a town and weather.
            # Data is a pkl file with vector of 7 on each line. 
            # Images has a tuple of RGB, Sem ground truth on each line.
            # Loop through the pkl files to get each record.

            file_name = str(town) + "_" + (str(weather).split('.'))[-1]

            with open('_out/' + file_name + '_data.pkl','wb') as of1:
                for measurement_tuple in measurements_list:
                    pickle.dump(measurement_tuple,of1)

            with open('_out/' + str(town) + '_images.pkl','wb') as of2:
                for image_tuple in images_list:
                    pickle.dump(image_tuple,of2)

            # Clear global buffers. 
            del measurements_list
            del images_list
            
