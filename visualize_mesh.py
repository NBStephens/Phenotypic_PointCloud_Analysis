'''



'''
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
sys.path.append(r"C:\Users\skk5802\Desktop\Phenotypic_PointCloud_Analysis")
from Code.visual_utils import *
from Code.pycpd_registrations_3D import consolidate_vtk, consolidate_case

directory = pathlib.Path(r"D:\Desktop\Carla\prenatal\results\stats")
os.chdir(directory)

#Set the theme, default,  dark,
pv.set_plot_theme("default")

rename_dict = {"__": "_"}


#Define the input name and read in to memory for visualization
mesh_name = "talus_consolidated_means.vtk"
input_mesh = pv.read(mesh_name)

input_mesh = scalar_name_cleanup(input_mesh=input_mesh, replace_dict=rename_dict)
inut_mesh.save(f"{mesh_name}")


#For getting the p-value thresholds in the same format.
stats_dir = pathlib.Path(r"Z:\RyanLab\Projects\SKuo\Medtool\medtool_training\new_point_cloud_cortical\results\stats\removed_gorilla_and_pongo\CtTh")
os.chdir(stats_dir)
stats_mesh = pv.read(glob.glob("*ttest.vtk")[0])
stats_mesh.save(str(glob.glob("*ttest.vtk")[0]))

#Output either a png or a pdf
get_scalar_screens(input_mesh=input_mesh,
                   scalars=["BVTV", "DA"],
                   limits=[0.0, 0.50],
                   consistent_limits=True,
                   n_of_bar_txt_portions=6,
                   output_type="pdf")

#If one scalar is being used or the range shoudl be somehting different
get_scalar_screens(input_mesh=input_mesh, scalars=["BVTV"], limits=[0.15, 0.25], consistent_limits=True, n_of_bar_txt_portions=6, output_type="png", from_bayes=True)
get_scalar_screens(input_mesh=input_mesh, scalars=["DA"], limits=[0.10, 0.30], consistent_limits=True, n_of_bar_txt_portions=6, output_type="png")
get_scalar_screens(input_mesh=input_mesh, scalars=["CtTh"], limits=False, consistent_limits=True, n_of_bar_txt_portions=6, output_type="pdf", from_bayes=True)


stats_mesh = scalar_name_cleanup(input_mesh=stats_mesh, replace_dict=rename_dict)

get_ttest_screens(input_mesh=stats_mesh,
                  scalars=["BVTV", "DA", "BSBV", "Tb_Sp", "Tb_Th"],
                  estimate_limits=True,
                  n_of_bar_txt_portions=11,
                  output_type="pdf")

get_ttest_screens(input_mesh=stats_mesh, scalars=["BVTV"], limits=[-6, 6], estimate_limits=False, output_type="png")
get_ttest_screens(input_mesh=stats_mesh, scalars=["DA"], limits=[-6, 6], estimate_limits=False, output_type="png")
get_ttest_screens(input_mesh=stats_mesh, scalars=["CtTh"], estimate_limits=True, n_of_bar_txt_portions=11, output_type="pdf")

#Clean up old arrays
for array in input_mesh.point_arrays:
    print(array)
    new_name = array.replace("Canis_", "")
    new_name = new_name.replace("C_", "get_rid_of")
    new_name = new_name.replace("canonical_", "Mean_")
    new_name = new_name.replace("canonincal_", "Mean_")
    new_name = new_name.replace("Neofelis_", "")
    input_mesh.rename_array(str(array), new_name)


#Gather all the pdfs into a single pdf
merge_pdfs(pdf_directory=stats_dir, string_match="", out_name="radius_cortical_results.pdf", output_directory="")

dist_plot_dir = r"Z:\RyanLab\Projects\SKuo\Medtool\medtool_training\new_point_cloud_cortical\results\stats\removed_gorilla_and_pongo\CtTh\Distplots"

merge_pdfs(pdf_directory=dist_plot_dir, string_match="", out_name="Canid_distplot_results.pdf", output_directory=dist_plot_dir)


#Bayes
directory = r"D:\Desktop\Sharon\visualize"
os.chdir(directory)

#Define the input name and read in to memory for visualization
mesh_name = "BSBV_smooth_bayes_HDI_range_mapped.vtk"
mesh_name = "max_normalized_CtTh_bayes_posterior_mean_mapped.vtk"
mesh_name = "max_normalized_CtTh_bayes_HDI_range_mapped.vtk"

bayes_outputs = ["posterior_mean", "posterior_sd", "HDI_range", "posterior_ninetyseven", "posterior_three"]
for output in bayes_outputs:
    consolidate_vtk(input_mesh=mesh_name, out_name=f"radius_{output}", name_match=f"{output}",
                    bayes_match=True, scalars=["BVTV", "DA", "BSBV", "Tb_Sp", "Tb_Th"], pairwise=False)

mesh_name = "radius_posterior_mean.vtk"
input_mesh = pv.read(mesh_name)

rename_dict = {"_nonKWapes": "_Homo_sapiens", "_KWapes": "_Pan_troglodytes", "_OWM": "_old_world_monkeys",
               "ESca1Mean_Radius_Dist_": "All_groups_", "All_groups_CtTh_": "CtTh_All_groups_", "_original_average": "",
               "max_normalized_CtTh_": "CtTh_max_normalized_", "CtTh_max_normalized_All_groups_mean": "CtTh_All_groups",
               "_std_": "_standard_dev_", "_standrd_dev_": "_standard_dev_"}

rename_dict = {"_std_": "_standard_dev_", "__": "_"}
rename_dict = {"CtTh_All_groups": "CtTh_max_normalized_All_groups", "__": "_"}
input_mesh = scalar_name_cleanup(input_mesh=input_mesh, replace_dict=rename_dict)
input_mesh = scalar_name_suffix(input_mesh=input_mesh, suffix="_posterior_HDI_range")
input_mesh = remove_array(input_mesh=input_mesh, remove_arrays=["All_groups"])

input_mesh.save(f"{mesh_name}")

get_scalar_screens(input_mesh=input_mesh,
                   scalars=["CtTh"],
                   limits=False,
                   consistent_limits=True,
                   n_of_bar_txt_portions=11,
                   output_type="pdf",
                   from_bayes=True,
                   scale_without_max_norm=False)

get_scalar_screens(input_mesh=input_mesh,
                   scalars=["BVTV", "DA", "BSBV", "Tb_Sp", "Tb_Th"],
                   limits=False,
                   consistent_limits=True,
                   n_of_bar_txt_portions=11,
                   output_type="pdf",
                   from_bayes=True)

mesh_name = "radius_posterior_sd.vtk"
input_mesh = pv.read(mesh_name)

rename_dict = {"_nonKWapes_": "_Homo_sapiens_", "_KWapes_": "_Pan_troglodytes_", "_OWM_": "_old_world_monkeys_"}
input_mesh = scalar_name_cleanup(input_mesh=input_mesh, replace_dict=rename_dict)

get_scalar_screens(input_mesh=input_mesh,
                   scalars=["BVTV", "DA", "BSBV", "Tb_Sp", "Tb_Th"],
                   limits=False,
                   consistent_limits=True,
                   n_of_bar_txt_portions=11,
                   output_type="pdf",
                   from_bayes=True)

mesh_name = "radius_HDI_range.vtk"
input_mesh = pv.read(mesh_name)

rename_dict = {"_nonKWapes_": "_Homo_sapiens_", "_KWapes_": "_Pan_troglodytes_", "_OWM_": "_old_world_monkeys_"}
input_mesh = scalar_name_cleanup(input_mesh=input_mesh, replace_dict=rename_dict)

get_scalar_screens(input_mesh=input_mesh,
                   scalars=["BVTV", "DA", "BSBV", "Tb_Sp", "Tb_Th"],
                   limits=False,
                   consistent_limits=True,
                   n_of_bar_txt_portions=11,
                   output_type="pdf",
                   from_bayes=True)

merge_pdfs(pdf_directory=directory, string_match="", out_name="Radius_trabecular_results.pdf", output_directory="")

#For getting the p-value thresholds in the same format.
stats_dir = pathlib.Path(r"D:\Desktop\Sharon\visualize\stats")
os.chdir(stats_dir)

bayes_stats = ["_POS_", "CohenD"]
for output in bayes_stats:
    consolidate_vtk(input_mesh="BSBV_smooth_bayes_CohenD_thresholded_map.vtk", out_name=f"radius_bayes_{output}_thresholds",
                    name_match=f"{output}", bayes_match=True, scalars=["BVTV", "DA", "BSBV", "Tb_Sp", "Tb_Th"], pairwise=True)

mesh_name = "max_normalized_CtTh_bayes_CohenD_thresholded_map.vtk"
stats_mesh = pv.read(f"{mesh_name}")
rename_dict = {"max_normalized_CtTh_": "CtTh_max_normalized_", "_nonKWapes": "_Homo_sapiens", "_KWapes": "_Pan_troglodytes_",
               "_OWM": "_old_world_monkeys", "__": "_", "_POS_": "_POS", "__": "_", "_Pan_troglodytes_":  "_Pan_troglodytes"}

stats_mesh = scalar_name_cleanup(input_mesh=stats_mesh, replace_dict=rename_dict)
stats_mesh = remove_array(input_mesh=stats_mesh, remove_arrays=["_original_average"])
stats_mesh = scalar_name_suffix(input_mesh=stats_mesh, suffix="_CohenD")
stats_mesh.save(f"{mesh_name}")

get_bayes_thresh_screens(input_mesh=stats_mesh, scalars=["CtTh"], limits=[0, 1],
                         estimate_limits=False, n_of_bar_txt_portions=11, output_type="pdf")

get_bayes_thresh_screens(input_mesh=stats_mesh, scalars=["Tb_Sp", "Tb_Th"], limits=[0, 1],
                         estimate_limits=False, n_of_bar_txt_portions=11, output_type="pdf")

stats_mesh = pv.read("CtTh_bayes_CohenD_thresholded_map.vtk")
rename_dict = {"_nonKWapes_": "_Homo_sapiens_", "_KWapes_": "_Pan_troglodytes_",
               "_OWM_": "_old_world_monkeys_", "__": "_", "_POS_": "_POS"}

stats_mesh = scalar_name_cleanup(input_mesh=stats_mesh, replace_dict=rename_dict)

get_bayes_thresh_screens(input_mesh=stats_mesh, scalars=["BVTV", "DA", "BSBV", "Tb_Sp", "Tb_Th"], limits=False,
                         estimate_limits=True, n_of_bar_txt_portions=11, output_type="pdf")

merge_pdfs(pdf_directory=directory, string_match="", out_name="Radius_trabecular_stats_results.pdf", output_directory="")

directory = r"D:\Desktop\Sharon\visualize\trab_pdf"
os.chdir(directory)

merge_pdfs(pdf_directory=directory, string_match="CtTh", out_name="Radius_cortical_results.pdf", output_directory="")


#For Rita

consolidate_case(inputMesh=, outName, nameMatch, scalars=["BVTV", "DA"], outputs=["_original_average.case", "_original_coef_var.case", "_original_standard_dev.case"], max_normazlied=True, pairwise=False)