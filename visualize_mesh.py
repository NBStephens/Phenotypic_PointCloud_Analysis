"""



"""
import shutil
import os
import re
import sys
import glob
import pathlib
import numpy as np
import pandas as pd
import pyvista as pv
from matplotlib import cm
from matplotlib.colors import rgb2hex
from PyPDF2 import PdfFileWriter, PdfFileReader
from scipy.spatial.transform import Rotation as R
from matplotlib.colors import LinearSegmentedColormap

sys.path.append(r"C:\Users\lvd5263\Desktop\Phenotypic_PointCloud_Analysis")

from Code.pycpd_registrations_3D import consolidate_vtk, consolidate_case
from Code.visual_utils import *

# script_dir = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))


########Get your screen shots############
bone = "tibia"
directory = pathlib.Path(
    rf"Z:\RyanLab\Projects\LDoershuk\diss_pointclouds\{bone}\results"
)
os.chdir(directory)

# Set the theme, default,  dark,
pv.set_plot_theme("dark")

rename_dict = {"__": "_"}

# Define the input name and read in to memory for visualization
mesh_name = f"consolidated_{bone}_means_transformed.vtk"
input_mesh = pv.read(mesh_name)

# This expects variable names to follow a convention.
rename_dict = {"_std": "_standard_dev", "__": "_", "_coef": "_coef_var", "_var_var": ""}
input_mesh = scalar_name_cleanup(input_mesh=input_mesh, replace_dict=rename_dict)
input_mesh.save(f"{mesh_name}")

# rename_dict = {"Th_standard_dev_max_norm": "Th_max_norm"}
# rename_dict = {"Sp_standard_dev_max_norm":"Sp_max_norm"}
# rename_dict = {"Sp_standard_dev_coef_var":"Sp_coef_var"}

max_norm_list = [item for item in list(input_mesh.point_arrays) if "max_norm" in item]
input_mesh = remove_array(input_mesh=input_mesh, remove_arrays=max_norm_list)

DA_norm_list = [
    item for item in list(input_mesh.point_arrays) if "DA_mean_value" in item
]
input_mesh = remove_array(input_mesh=input_mesh, remove_arrays=DA_norm_list)
input_mesh.save(f"{mesh_name}")

# Output either a png or a pdf
get_scalar_screens(
    input_mesh=input_mesh,
    scalars=["BVTV"],
    limits=[0.0, 0.4],
    consistent_limits=True,
    n_of_bar_txt_portions=6,
    output_type="png",
    from_bayes=False,
    scale_without_max_norm=True,
    foot_bones=False,
)

# If one scalar is being used or the range should be something different
get_scalar_screens(
    input_mesh=input_mesh,
    scalars=["BVTV"],
    limits=[0.0, 0.55],
    consistent_limits=True,
    n_of_bar_txt_portions=6,
    output_type="png",
    from_bayes=False,
    scale_without_max_norm=True,
    foot_bones=False,
)


get_scalar_screens(
    input_mesh=input_mesh,
    scalars=["DA"],
    limits=[0.0, 0.45],
    consistent_limits=True,
    n_of_bar_txt_portions=6,
    output_type="png",
    from_bayes=False,
    scale_without_max_norm=True,
)

get_scalar_screens(
    input_mesh=input_mesh,
    scalars=["CtTh"],
    limits=False,
    consistent_limits=True,
    n_of_bar_txt_portions=6,
    output_type="pdf",
    from_bayes=True,
)

# For getting the p-value thresholds in the same format.
# stats_dir = directory.joinpath("stats")
stats_dir = directory.parent.joinpath("visualize")
os.chdir(stats_dir)
stats_mesh = pv.read(stats_dir.joinpath("talus_t_tests_analysis_3.vtk"))
max_norm_list = [item for item in list(stats_mesh.point_arrays) if "max_norm" in item]
stats_mesh = remove_array(input_mesh=stats_mesh, remove_arrays=max_norm_list)

stats_mesh = scalar_name_cleanup(input_mesh=stats_mesh, replace_dict=rename_dict)

get_ttest_screens(
    input_mesh=stats_mesh,
    scalars=["BVTV", "DA"],  # , "BSBV", "Tb_Sp", "Tb_Th"],
    limits=[-5.0, 5.0],
    estimate_limits=False,
    n_of_bar_txt_portions=11,
    output_type="pdf",
)

get_ttest_screens(
    input_mesh=stats_mesh,
    scalars=["BVTV", "DA"],  # , "BSBV", "Tb_Sp", "Tb_Th"],
    limits=[-100.0, 100.0],
    estimate_limits=False,
    n_of_bar_txt_portions=11,
    output_type="png",
)

get_ttest_screens(
    input_mesh=stats_mesh,
    scalars=["BVTV"],
    limits=[-6, 6],
    estimate_limits=False,
    output_type="png",
)
get_ttest_screens(
    input_mesh=stats_mesh,
    scalars=["DA"],
    limits=[-6, 6],
    estimate_limits=False,
    output_type="png",
)
get_ttest_screens(
    input_mesh=stats_mesh,
    scalars=["CtTh"],
    estimate_limits=True,
    n_of_bar_txt_portions=11,
    output_type="pdf",
)


# Clean up old arrays
for array in input_mesh.point_arrays:
    print(array)
    new_name = array.replace("Canis_", "")
    new_name = new_name.replace("C_", "get_rid_of")
    new_name = new_name.replace("canonical_", "Mean_")
    new_name = new_name.replace("canonincal_", "Mean_")
    new_name = new_name.replace("Neofelis_", "")
    input_mesh.rename_array(str(array), new_name)


# Gather all the pdfs into a single pdf
merge_pdfs(
    pdf_directory=stats_dir,
    string_match="",
    out_name="radius_cortical_results.pdf",
    output_directory="",
)

dist_plot_dir = r"Z:\RyanLab\Projects\SKuo\Medtool\medtool_training\new_point_cloud_cortical\results\stats\removed_gorilla_and_pongo\CtTh\Distplots"

merge_pdfs(
    pdf_directory=dist_plot_dir,
    string_match="",
    out_name="Canid_distplot_results.pdf",
    output_directory=dist_plot_dir,
)

#######gather the vtks##########

# Where all the single vtk files are
stats_folder = pathlib.Path(
    r"Z:\RyanLab\Projects\LDoershuk\diss_pointclouds\tibia\results\stats"
)

# The scalars you are statsing on
# scalar_list = ['BVTV', 'DA', 'Tb_Sp', 'Tb_Th', 'Tb_N']

# The output for the vtk files
vtk_out_dir = pathlib.Path(
    r"Z:\RyanLab\Projects\LDoershuk\diss_pointclouds\tibia\visualize"
)

# !!!!!!! THIS WILL NOT WORK WITHOUT SOME SORT OF LIST!

os.chdir(stats_folder)
current_folder_list = stats_folder.glob("*_traces")
for folders in current_folder_list:
    vtk_folders = stats_folder.joinpath(folders)
    vtk_files = [x for x in vtk_folders.rglob("*.vtk")]
    for vtk_file in vtk_files:
        shutil.copy(vtk_file, vtk_out_dir)

os.chdir(vtk_out_dir)

# Bayes directory
directory = pathlib.Path(
    r"Z:\RyanLab\Projects\LDoershuk\diss_pointclouds\calcaneus\visualize"
)
os.chdir(directory)

# Define the input name and read in to memory for visualization
bone = "calcaneus"
mesh_name = f"consolidated_{bone}_means_transformed.vtk"


# Gather up all the results from the different scalars
bayes_outputs = [
    "posterior_mean",
    "posterior_sd",
    "HDI_range",
    "posterior_ninetyseven",
    "posterior_three",
]

# this loop will put them into a single vtk for each output in the list. It does expect names to follow a convention
for output in bayes_outputs:
    consolidate_vtk(
        input_mesh=mesh_name,
        out_name=f"{bone}_{output}",
        name_match=f"{output}",
        bayes_match=True,
        scalars=["BVTV", "DA", "Tb_N", "Tb_Sp", "Tb_Th"],
        pairwise=False,
        missing_scalar_name=True,
    )  # this is if the scalar wasn't written to the array

# This is to get the various maps
mesh_name = f"{bone}_posterior_mean.vtk"
input_mesh = pv.read(mesh_name)

# This is to clean up any names you don't want or ugly conventions
# The functions do expect them to have a certain format for labels
rename_dict = {
    "_nonKWapes": "_Homo_sapiens",
    "_KWapes": "_Pan_troglodytes",
    "_OWM": "_old_world_monkeys",
    "ESca1Mean_Radius_Dist_": "All_groups_",
    "All_groups_CtTh_": "CtTh_All_groups_",
    "_original_average": "",
    "max_normalized_CtTh_": "CtTh_max_normalized_",
    "CtTh_max_normalized_All_groups_mean": "CtTh_All_groups",
    "_std_": "_standard_dev_",
    "_standrd_dev_": "_standard_dev_",
}

# rename_dict = {"_std": "_standard_dev", "__": "_", "_coef": "_coef_var"}
# rename_dict = {"CtTh_All_groups": "CtTh_max_normalized_All_groups", "__": "_"}
# This does the clean up based on the key value pairs defined in the dict. Repeat until they look good: input_mesh.points_arrays
input_mesh = scalar_name_cleanup(input_mesh=input_mesh, replace_dict=rename_dict)
input_mesh = scalar_name_suffix(input_mesh=input_mesh, suffix="_posterior_HDI_range")

# This will remove an array that you don't wnat
input_mesh = remove_array(input_mesh=input_mesh, remove_arrays=["All_groups"])

# Save it when they are clean and reorient it in paraview if needed
input_mesh.save(f"{mesh_name}")

# For a single scalar
get_scalar_screens(
    input_mesh=input_mesh,
    scalars=["BVTV"],
    limits=False,
    consistent_limits=True,
    n_of_bar_txt_portions=11,
    output_type="png",
    from_bayes=True,
    scale_without_max_norm=False,
    foot_bones=True,
)

# For all the scalars with estimated limits
get_scalar_screens(
    input_mesh=input_mesh,
    scalars=["BVTV", "DA", "Tb_N", "Tb_Sp", "Tb_Th"],
    limits=False,
    consistent_limits=True,
    n_of_bar_txt_portions=11,
    output_type="png",
    from_bayes=True,
)

# Repeat with the other meshes if you want
mesh_name = f"{bone}_posterior_sd.vtk"
input_mesh = pv.read(mesh_name)

rename_dict = {
    "_nonKWapes_": "_Homo_sapiens_",
    "_KWapes_": "_Pan_troglodytes_",
    "_OWM_": "_old_world_monkeys_",
}

input_mesh = scalar_name_cleanup(input_mesh=input_mesh, replace_dict=rename_dict)

get_scalar_screens(
    input_mesh=input_mesh,
    scalars=["BVTV", "DA", "BSBV", "Tb_Sp", "Tb_Th"],
    limits=False,
    consistent_limits=True,
    n_of_bar_txt_portions=11,
    output_type="pdf",
    from_bayes=True,
)

mesh_name = f"{bone}_HDI_range.vtk"
input_mesh = pv.read(mesh_name)

rename_dict = {
    "_nonKWapes_": "_Homo_sapiens_",
    "_KWapes_": "_Pan_troglodytes_",
    "_OWM_": "_old_world_monkeys_",
}
input_mesh = scalar_name_cleanup(input_mesh=input_mesh, replace_dict=rename_dict)

get_scalar_screens(
    input_mesh=input_mesh,
    scalars=["BVTV", "DA", "BSBV", "Tb_Sp", "Tb_Th"],
    limits=False,
    consistent_limits=True,
    n_of_bar_txt_portions=11,
    output_type="pdf",
    from_bayes=True,
)

merge_pdfs(
    pdf_directory=directory,
    string_match="",
    out_name=f"{bone}_trabecular_results.pdf",
    output_directory="",
)

# For getting the p-value thresholds in the same format.
# stats_dir = pathlib.Path(r"D:\Desktop\Sharon\visualize\stats")
# os.chdir(stats_dir)

# Consolidate stats meshes
bayes_stats = ["_POS_", "CohenD"]
for output in bayes_stats:
    consolidate_vtk(
        input_mesh=f"consolidated_{bone}_means_transformed.vtk",
        out_name=f"{bone}_bayes_{output}_thresholds",
        name_match=f"{output}",
        bayes_match=True,
        scalars=["BVTV", "DA", "Tb_N", "Tb_Sp", "Tb_Th"],
        pairwise=True,
        missing_scalar_name=True,
    )

mesh_name = f"{bone}_bayes_CohenD_thresholds.vtk"
stats_mesh = pv.read(f"{mesh_name}")

# Clean it up if needed
rename_dict = {
    "max_normalized_CtTh_": "CtTh_max_normalized_",
    "_nonKWapes": "_Homo_sapiens",
    "_KWapes": "_Pan_troglodytes_",
    "_OWM": "_old_world_monkeys",
    "__": "_",
    "_POS_": "_POS",
    "__": "_",
    "_Pan_troglodytes_": "_Pan_troglodytes",
}

stats_mesh = scalar_name_cleanup(input_mesh=stats_mesh, replace_dict=rename_dict)
stats_mesh = remove_array(input_mesh=stats_mesh, remove_arrays=["_original_average"])
stats_mesh = scalar_name_suffix(input_mesh=stats_mesh, suffix="_CohenD")

# Save the cleaned one and then reorient. Makes sure you keep paraview on from the last reorinet so you can copy and paste the transform!
stats_mesh.save(f"{mesh_name}")

get_bayes_thresh_screens(
    input_mesh=stats_mesh,
    scalars=["DA"],
    limits=[-2, 2],
    estimate_limits=False,
    n_of_bar_txt_portions=11,
    output_type="png",
    foot_bones=True,
)

get_bayes_thresh_screens(
    input_mesh=stats_mesh,
    scalars=["Tb_Sp", "Tb_Th"],
    limits=[0, 1],
    estimate_limits=False,
    n_of_bar_txt_portions=11,
    output_type="pdf",
    foot_bones=True,
)

stats_mesh = pv.read(f"{bone}_bayes_POS_thresholds.vtk")
rename_dict = {
    "_nonKWapes_": "_Homo_sapiens_",
    "_KWapes_": "_Pan_troglodytes_",
    "_OWM_": "_old_world_monkeys_",
    "__": "_",
    "_POS_": "_POS",
}

stats_mesh = scalar_name_cleanup(input_mesh=stats_mesh, replace_dict=rename_dict)

get_bayes_thresh_screens(
    input_mesh=stats_mesh,
    scalars=["BVTV", "DA", "Tb_N", "Tb_Sp", "Tb_Th"],
    limits=[-2.0, 2.0],
    estimate_limits=True,
    n_of_bar_txt_portions=11,
    output_type="png",
)

merge_pdfs(
    pdf_directory=directory,
    string_match="",
    out_name="Radius_trabecular_stats_results.pdf",
    output_directory="",
)

directory = directory
os.chdir(directory)
