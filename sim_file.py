"""
_________________________________________________________________________________________________
                                                                                                 |
Authors: * Ulrich Prestel    <Ulrich.Prestel@protonmail.com>                                     |
_________________________________________________________________________________________________|
"""
import h5py
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

class HDF5Dataset():
    SIM_DATA = ['sdf_obstacles', 'emitter_value']
    SIM_ATTR = ['dt', 'resolution_x', 'resolution_y', 'sim_time', 'finished']
    FRAME_DATA = ['velocity_x', 'velocity_y', 'pressure', 'density']

    # reload file to update the dataset-length and available simulations
    def reload(self):
        self.datasets = {}
        self.resolution_x = 0
        self.resolution_y = 0

        # go over all simulations and add them to the index
        for sim_name in self.hdf5_file.keys():
            sim = self.hdf5_file[sim_name]
            # check for validity of the simulation
            if not 'sim ' in sim.name: # all simulations start with sim
                continue
            if not sim.attrs['finished']: # only finished simulations should be used
                continue
            if sim.attrs['resolution_x'] <= 0 or sim.attrs['resolution_y'] <= 0: # the resolution is only valide if >0
                continue
            if len(sim['velocity_x']) != len(sim['velocity_y']) != len(sim['pressure']) != len(sim['density']): # the number of frames must be the same everywhere
                continue
            if self.only != "" and sim_name != self.only: # used if only one simulation should be used (see only_use)
                continue

            # set resolution if not already set
            if self.resolution_x == 0 and self.resolution_y == 0:
                self.resolution_x = sim.attrs['resolution_x']
                self.resolution_y = sim.attrs['resolution_y']

            # ensure same resolution only use simulations of same resolution if flag is set
            if self.same_resolution_size and (sim.attrs['resolution_x'] != self.resolution_x or
                                              sim.attrs['resolution_y'] != self.resolution_y):
                continue

            self.datasets[sim.name] = len(sim['pressure']) - self.consecutive_frames # save number of usable indizes per simulation

        # sum up usable indizes
        self.dataset_len = 0
        for simulation, frame_count in self.datasets.items():
            self.dataset_len += frame_count

        # list used to map dataset-indizes to simulation and frame (see map_id_to_frame)
        self.simulation_order = sorted(list(self.datasets.keys()))


    def __init__(self, data_file, mode='r', consecutive_frames=1,
                 same_resolution_size=True, sim_data=SIM_DATA, frame_data=FRAME_DATA,
                 sim_attr=SIM_ATTR, only="", return_transform=lambda a: a):

        # open data file
        self.hdf5_file = h5py.File(data_file, mode, libver='earliest')

        # number of consecutive frames to return
        self.consecutive_frames = consecutive_frames
        # only use simulations with same resolution
        self.same_resolution_size = same_resolution_size

        # used to limit to one simulation
        self.only = only

        # information about the data
        self.sim_data = sim_data
        self.sim_attr = sim_attr
        self.frame_data = frame_data

        # the transform used on the return data
        self.return_transform = return_transform

        # make sure the file contains an id for the next simulation
        try:
            next_id = self.hdf5_file.attrs['next_id']
        except KeyError as e:
            if mode != 'r':
                self.hdf5_file.attrs['next_id'] = len(self.hdf5_file.keys()) + 1

        # update/initialize simulation-list and dataset-length
        self.reload()

    # clean-up functions
    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    # limit dataset to only one simulation
    def only_use(self, sim_name=""):
        self.only = sim_name
        self.reload()

    # find the corresponding simulation for an integer index
    def map_id_to_frame(self, index):
        # find the simulation by subtracting the available indizes per simulation in order given by simulation_order
        simulation_index = 0
        while index > self.datasets[self.simulation_order[simulation_index]]:
            index -= self.datasets[self.simulation_order[simulation_index]] # subtract possible indizes from the index to check in the next simulation
            simulation_index += 1

        # if the index is in the simulation-index-count the simulation-index is found and the remainder is the frame index
        return self.simulation_order[simulation_index], index

    # return an element based on:
    #   string to get an HDF5-Entry
    #   a tuple to directly pass simulation and frame index
    #   an int for easy iteration
    def __getitem__(self, index):
        # check for string index
        if type(index) is str:
            return self.hdf5_file[index]

        # check for direct entry of both indizes if only one integer given calculate corresponding indizes
        if type(index) is tuple:
            simulation_name, index = index
        else:
            simulation_name, index = self.map_id_to_frame(index)

        # get the simulation entry for ease of use
        simulation = self.hdf5_file[simulation_name]

        # copy all data as given in __init__ into a dictionary
        data = {}
        # copy single-matrix-data (sdf)
        for key in self.sim_data:
            data[key] = simulation[key][:]
        # copy attributes (resolution)
        for key in self.sim_attr:
            data[key] = simulation.attrs[key]
        # copy frame data (speed)
        for key in self.frame_data:
            data[key] = simulation[key][index:index + self.consecutive_frames] # slice according to consecutive_frames

        return self.return_transform(data)

    # return length as determined in reload()
    def __len__(self):
        return self.dataset_len

    # add a simulation entry and create frame-datasets
    def add_simulation(self, sim_attr, sim_data):

        # add simulation with new id
        sim = self.hdf5_file.create_group("sim {:03}".format(self.hdf5_file.attrs['next_id']))
        # update next_id
        self.hdf5_file.attrs['next_id'] += 1

        # set internal resolution if not already set
        if self.resolution_x == 0 or self.resolution_y == 0:
            self.resolution_x = sim_attr['resolution_x']
            self.resolution_y = sim_attr['resolution_y']

        # save all passed attributes
        for key, data in sim_attr.items():
            sim.attrs[key] = data

        # enforce time and finished attributes to defaults
        sim.attrs['sim_time'] = 0
        sim.attrs['finished'] = False

        # create datasets for the keys given in __init__
        for key, data in sim_data.items():
            sim.create_dataset(key, (self.resolution_x, self.resolution_y), data=data)

        for key in self.frame_data:
            sim.create_dataset(key, (0, self.resolution_x, self.resolution_y),
                               chunks=(1, self.resolution_x, self.resolution_y),
                               maxshape=(None, self.resolution_x, self.resolution_y), compression='gzip',
                               compression_opts=4) # the chunks are needed for compression

        return sim.name

    # add a single frame to a simulation
    def add_frame(self, sim_name, frame_data):
        sim = self.hdf5_file[sim_name]
        # only edit non finished simulations
        if sim.attrs['finished']:
            raise RuntimeError("simulation is marked finished!")

        # prepare the dataset to hold one more frame
        # get new shape
        shape = sim[self.frame_data[0]].shape

        shape = (shape[0] + 1, shape[1], shape[2])
        index = shape[0] - 1

        # reshape datasets
        for key in self.frame_data:
            sim[key].resize(shape)

        # using keys from self.frame_data ensures all (and only) configured fields
        # are saved
        for key in self.frame_data:
            sim[key][index] = frame_data[key]

    # mark a simulation as finished so it is used with integer indexing and read-only
    def mark_finished(self, sim_name, time, sim_attr={}):
        sim = self.hdf5_file[sim_name]

        # alrady finished
        if sim.attrs['finished']:
            raise RuntimeError("simulation is marked finished!")

        sim.attrs['finished'] = True
        sim.attrs['sim_time'] = time # save needed time

        for key, attr in sim_attr.items():
            sim.attrs[key] = attr

        self.reload() # aadjust dataset-length, and simulation-list so the new simulation can be indexed

    # wrapper to get the simulation names
    def get_simulation_names(self):
        return list(self.hdf5_file.keys())

    # wrapper to get the number of frames in a simulation
    def get_number_of_frames(self, sim_name):
        return len(self.hdf5_file[sim_name]['pressure'])

    # passthrough close to HDF5-file
    def close(self):
        self.hdf5_file.close()

        
def split_data(sfile, validation_split = .2, batch_size=4):

    dataset_size = len(sfile)
    indices = list(range(dataset_size))

    split = int(np.floor(validation_split * dataset_size))
    train_indices = indices[split:]
    val_indices = indices[:split]

    # we create random samplers on our subsets.
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    print(dataset_size)

    train_loader = torch.utils.data.DataLoader(sfile, batch_size=batch_size,
                                        num_workers=1,
                                        sampler=train_sampler)

    validation_loader = torch.utils.data.DataLoader(sfile, batch_size=batch_size,
                                        num_workers=1,
                                        sampler=val_sampler)

    #print(len(validation_loader))

    return train_loader, validation_loader        
