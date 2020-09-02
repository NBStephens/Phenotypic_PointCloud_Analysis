'''
To get nice looking distplaots for populations

'''
import os
import sys
import glob
import pathlib
import pandas as pd
sys.path.append(r"C:\Users\nbs49\Desktop\git_pulls\Phenotypic_PointCloud_Analysis")
from Code.pycpd_registrations_3D import get_distplot

#Define a custom color palette
color_pallete = ["#72008F", "#CC0000", "#005EBA", "#000000"] #FFB90F"], #"#00C4CC"]

#Point towrads the poitn cloud directory and change into it
point_cloud_dir = pathlib.Path(r"D:\Desktop\Canids\Point_cloud\Humerus")
os.chdir(point_cloud_dir)

#Provide the sclar list, which can be quite long.
scalar_list = ["BVTV"]#, "DA"]
#scalar_list = ["CtTh"]
bone = "Humerus"
#Grab the data from all groups in the previous step and extend the empty list
scalars = []
for scalar in scalar_list:
    search_item = point_cloud_dir.joinpath(scalar).joinpath(f"Data_from_all_groups_{scalar}.csv")
    scalars.extend(glob.glob(str(search_item)))
    #remove_string = "_max_normalized.csv"
    #scalar_list = [item for item in scalar_list if remove_string not in item]
    #remove_string = "_voxel.csv"
    #scalar_list = [item for item in scalar_list if remove_string not in item]
    #scalar_list.sort()

#Loop through the scalars list and create a distplot for each
for scale in scalars:
    scalar = pathlib.Path(scale).parts[-1]
    scalar = scalar.replace(".csv", "")
    scalar = scalar.replace("Data_from_all_groups_", "")
    print(scalar)
    registered_dataset = pd.read_csv(scale, index_col=None)
    xlim_min = registered_dataset.min(axis=1).min() - registered_dataset.min(axis=1).min() * .10
    xlim_max = registered_dataset.max(axis=1).max() + registered_dataset.max(axis=1).max() * .10
    xlim_max = 0.75
    dist_plot = get_distplot(registered_dataset=registered_dataset,
                             scalar=f"{bone} {scalar}",
                             fontsize=3,
                             legendfont=30,
                             xlim=[xlim_min, xlim_max],
                             colors=color_pallete)
    save_name = point_cloud_dir.joinpath(scalar).joinpath(f"{bone}_{scalar}_dist_plot.tif")
    dist_plot.savefig(str(save_name), dpi=600)
    dist_plot.savefig(f"{str(save_name)[:-4]}.pdf", dpi=600)

    #If you've produced the max normalized meshes you can use this
    point_data = f"Data_from_all_groups_max_normalized_{scale}.csv"
    point_data = glob.glob(str(pathlib.Path(point_cloud_dir).joinpath(scale).joinpath(point_data)))
    registered_dataset = pd.read_csv(point_data[0], index_col=None)
    dist_plot = get_distplot(registered_dataset=registered_dataset,
                             scalar=scale + "_max_normalized",
                             fontsize=3,
                             legendfont=30,
                             xlim=1.0,
                             colors=color_pallete)
    save_name = pathlib.Path(point_cloud_dir).joinpath(scale).joinpath(f"{scale}_max_normalized_dist_plot.tif")
    dist_plot.savefig(str(save_name), dpi=600)
