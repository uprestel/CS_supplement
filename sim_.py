# coding=utf-8
"""
_________________________________________________________________________________________________
                                                                                                 |
Authors: * Ulrich Prestel    <Ulrich.Prestel@protonmail.com>                                     |
       : * Holger Wünsche    <Holger.o.wuensche@t-online.de>                                     |
_________________________________________________________________________________________________|
"""

import math
import os
import shutil
import sys
import time
import numpy as np
from manta import *
import sim_file
import sim_geo

if len(sys.argv) != 5:
    print("ERROR: not enough arguments! \nusage: steps filename geometrypath number_of_obstacles")
    sys.exit()

steps = int(sys.argv[1])
filename = sys.argv[2]
geometryPath = "{}/*.png".format(sys.argv[3])
n_obstacles = int(sys.argv[4])

# ------------------------- parameters ------------------------- #
#steps = 10
savedata = True
basePath = "../data/"
#geometryPath = "objs/*.png"
#filename = "sim_file.h5"
timestep = 0.5
showGui = False  # show UI
dim = 2
res_x = 64
res_y = 64  # 92
bWidth = 1
n_emitters = 1
#n_obstacles = 10



# ------------------------- scene settings ------------------------- #
setDebugLevel(0)

# ------------------------- solver params ------------------------- #

# The grid size
gs = vec3(res_x, res_y, 1)
# The domain in [xmin, xmax, ymin, ymax]
dom = [bWidth, res_x - bWidth, bWidth, res_y - bWidth]
XMIN, XMAX, YMIN, YMAX = dom
buoy = vec3(0, -1e-3, 0)

# wlt Turbulence input fluid
sm = Solver(name='smaller', gridSize=gs, dim=dim)
sm.timestep = timestep

timings = Timings()

# ------------------------- simulation grids ------------------------- #
flags = sm.create(FlagGrid)
vel = sm.create(MACGrid)
density = sm.create(RealGrid)
pressure = sm.create(RealGrid)
obstacleGrid = sm.create(LevelsetGrid)
obstacleGrid2 = sm.create(LevelsetGrid)
emitterGrid = sm.create(LevelsetGrid)

# we open the boundaries at the top and bottom, so the smoke can escape.

flags.initDomain(boundaryWidth=bWidth)
setObstacleFlags(flags=flags, phiObs=obstacleGrid)
flags.fillGrid()

setOpenBound(flags, bWidth, 'yY', FlagOutflow | FlagEmpty)

# ------------------------- inflow sources ------------------------- #
# list of the emitters in the scene. Currently they are represented as the following tuples:
# (Shape of the object, vec_vel, value_den)
# * value_den should be between [-1,1]. If it is >=0, the smoke will rise. If it is < 0, the smoke
#           will fall down.
# * vec_vel should have the magnitude of 1.

# sources = []
# for _ in range(0, n_emitters):
#     ang = np.random.randint(low=0, high=360)
#     ang_rad = math.radians(ang)
#     vec_vel = vec3(math.cos(ang_rad), math.sin(ang_rad), 0)
#     value_den = np.random.uniform(low=-1, high=1)
#     radius, center = sim_geo.generate_random_circle(dom=dom)
#     # sources.append((sm.create(Sphere, center=gs * vec3(0.5, 0.1, 0.5), radius=res_x * 0.14 / 2), 1, 1))
#     sources.append((sm.create(Sphere, center=center, radius=radius), 1, 1))

# FIXED SETUP: FIXED EMITTER FOR TESTING.
# the emitters are represented by the following tuple: (shape, radius, density_value)
sources = []
sources.append((sm.create(Sphere, center=gs * vec3(0.5, 0.1, 0.5), radius=res_x * 0.14 / 2), res_x * 0.14 / 2, 1))
for source_t in sources:
    source, radius, value_den = source_t
    source.applyToGrid(grid=emitterGrid, value=value_den)

np_emitter = np.zeros((res_x, res_y, 1), dtype="f")
copyGridToArrayReal(source=emitterGrid, target=np_emitter)
# ------------------------- obstacle creation ------------------------- #
# obs = Sphere( parent=sm,   center=gs*vec3(0.5,0.2,0.5), radius=res_x*0.15)
# dom2 = [XMIN + res_x / 4, XMAX - res_x / 4, YMIN + res_y / 3, YMAX]
# radius, center = sim_geo.generate_random_circle(dom=dom2)

geogen = sim_geo.GeometryGenerator(geopath=geometryPath, obstacleGrid=obstacleGrid2, parent=sm, flags=flags,
                                   res_x=res_x,
                                   res_y=res_y,
                                   FlagObstacle=FlagObstacle, emitters=sources)
geogen.generate(n_obstacles)

np_sdf = np.zeros((res_x, res_y, 1), dtype="f")
copyGridToArrayReal(source=obstacleGrid2, target=np_sdf)
np_sdf = sim_geo.generate_sdf_field(np_sdf, res_x, res_y)

# ------------------------- GUI ------------------------- #
if showGui and GUI:
    gui = Gui()
    gui.show()
    gui.pause()

# ------------------------- main loop ------------------------- #
t = 0
with sim_file.HDF5Dataset(basePath + filename, "a") as simfile:
    sim_attr = {
        "dt": timestep,
        "resolution_x": res_x,
        "resolution_y": res_y
    }
    sim_data = {
        "sdf_obstacles": np_sdf,
        "emitter_value": np_emitter
    }
    if savedata:
        sim_name = simfile.add_simulation(sim_attr, sim_data)
        sim_start_time = time.time()
    while t < steps:
        curt = t * sm.timestep
        for source_t in sources:
            source, value_vel, value_den = source_t
            source.applyToGrid(grid=density, value=value_den)
            # source.applyToGrid(grid=vel, value=vec3(0, 1.7, 0))
            source.applyToGrid(grid=emitterGrid, value=1)
        #mantaMsg("Current time t: " + str(curt) + " \n")

        advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2, openBounds=True, boundaryWidth=bWidth)
        advectSemiLagrange(flags=flags, vel=vel, grid=vel, order=2, openBounds=True, boundaryWidth=bWidth)
        setWallBcs(flags=flags, vel=vel)
        addBuoyancy(density=density, vel=vel, gravity=buoy, flags=flags)

        vorticityConfinement(vel=vel, flags=flags, strength=0.05)
        solvePressure(flags=flags, vel=vel, pressure=pressure, cgMaxIterFac=10.0, cgAccuracy=0.0001)
        setWallBcs(flags=flags, vel=vel)

        # --- saving to np array --- #
        if savedata:
            np_pressure = np.zeros((res_x, res_y, 1), dtype="f")
            copyGridToArrayReal(source=pressure, target=np_pressure)

            np_velocity = np.zeros((res_x, res_y, 3), dtype="f")
            copyGridToArrayVec3(source=vel, target=np_velocity)

            np_density = np.zeros((res_x, res_y, 1), dtype="f")
            copyGridToArrayReal(source=density, target=np_density)

            # mantaMsg("SHAPE: %s" % str(np_velocity.shape)+ str( np_velocity[:, :, 1].shape))
            frame_data = {
                "velocity_x": np_velocity[:, :, 0],
                "velocity_y": np_velocity[:, :, 1],
                "pressure": np_pressure[:, :, 0],
                "density": np_density[:, :, 0]
            }

            simfile.add_frame(sim_name, frame_data)

        sm.step()
        t = t + 1
    if savedata:
        sim_end_time = time.time()
        simfile.mark_finished(sim_name, sim_end_time-sim_start_time)



    simfile.close()
