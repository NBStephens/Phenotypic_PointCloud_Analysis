import os
import sys
import pathlib
import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap
from scipy.spatial.transform import Rotation as R
from matplotlib.colors import LinearSegmentedColormap

directory = pathlib.Path(r"D:\Desktop\Sharon\visualize")
os.chdir(directory)

#Set the "Dark" theme
pv.set_plot_theme("night")

#Set up custom colormap from paraview
rainbow_desat = LinearSegmentedColormap.from_list('mycmap', ['#4747db', '#00005c', '#00ffff', '#008000', '#ffff00', '#ff6100', '#6b0000', '#e04d4d'])

#Define the input name and read in to memory for visualization
inputMesh = "BVTV_smooth_bayes_posterior_mean_mapped.vtk"
mesh = pv.read(inputMesh)

#Midslices
slc_x = mesh.slice(normal='x')
slc_y = mesh.slice(normal='y')
slc_z = mesh.slice(normal='z')

#Set up a name for the screen shot
screen_shot = f"{inputMesh[:-4]}.png"
screen_shot = screen_shot.replace("..", ",")

#Set up the arguments to be passed to pyvista
vel_dargs = dict(scalars="BVTV_smooth_mean_Anteaters", clim=[0, .50], cmap=rainbow_desat,
                 log_scale=False, reset_camera=False)
f_size = 15

sargs = dict(title_font_size=1, label_font_size=25, shadow=False, n_labels=6, vertical=True,
             position_x=0.05, position_y=0.2,
             height=0.65, width=0.1, italic=False, fmt="%.2f", font_family="arial")

plotter = pv.Plotter(shape=(3, 3), window_size=(3840, 2160))
plotter.disable_parallel_projection()
plotter.reset_camera()

plotter.subplot(0, 0)
plotter.add_mesh(mesh, **vel_dargs, scalar_bar_args=sargs)
plotter.view_zy()
current_pos = plotter.camera_position
print(current_pos)
new_position = [current_pos[0], current_pos[1], (0.0, 0.0, 1.0)]
print(new_position)
plotter.camera_position = new_position
plotter.add_text("Dorsal", font_size=f_size)

plotter.subplot(1, 0)
plotter.add_mesh(mesh, **vel_dargs)
plotter.view_yz()
plotter.add_text("Anterior", font_size=f_size)


plotter.subplot(0, 1)
plotter.add_mesh(mesh, **vel_dargs)
plotter.view_zx()
current_pos = plotter.camera_position
print(current_pos)
r = R.from_euler('xyz', [0.0, 90.0, 0.0], degrees=True)
new_position = tuple(np.dot(np.array(r.as_matrix()), np.array(current_pos[0])))
plotter.camera_position = [new_position, current_pos[1], (0.0, 1.0, 0.0)]
print(plotter.camera_position)
plotter.add_text("Medial", font_size=f_size)

plotter.subplot(1, 1)
plotter.add_mesh(mesh, **vel_dargs)
plotter.view_xz()
plotter.add_text("Lateral", font_size=f_size)


plotter.subplot(0, 2)
plotter.add_mesh(mesh, **vel_dargs)
plotter.view_yx()
plotter.add_text("Distal", font_size=f_size)
plotter.add_text("mean BV/TV Anteater", position='upper_right', font_size=25,
                 font=None, shadow=False, name=None, viewport=False)


plotter.subplot(1, 2)
plotter.add_mesh(mesh, **vel_dargs)
plotter.view_yx()
current_pos = plotter.camera_position
print(current_pos)
r = R.from_euler('xyz', [0.0, 180.0, 0.0], degrees=True)
new_position = tuple(np.dot(np.array(r.as_matrix()), np.array(current_pos[0])))
new_position = [new_position, current_pos[1], current_pos[2]]
print(new_position)
plotter.camera_position = new_position
plotter.add_text("Proximal", font_size=f_size)

plotter.reset_camera()

plotter.subplot(2, 0)
plotter.add_mesh(slc_x, **vel_dargs, scalar_bar_args=sargs)
plotter.view_yz()
plotter.add_text("X-midplane", font_size=f_size)

plotter.subplot(2, 1)
plotter.add_mesh(slc_y, **vel_dargs)
plotter.view_xz()
plotter.add_text("Y-midplane", font_size=f_size)

plotter.subplot(2, 2)
plotter.add_mesh(slc_z, **vel_dargs)
plotter.view_yx()
plotter.add_text("Z-midplane", font_size=f_size)

plotter.show(screenshot=str(screen_shot))

