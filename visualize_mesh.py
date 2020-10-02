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
sys.path.append(r"D:\git_pulls\Phenotypic_PointCloud_Analysis")
from Code.visual_utils import *
from Code.pycpd_registrations_3D import consolidate_vtk

directory = pathlib.Path(r"D:\Desktop\Sharon\visualize")
os.chdir(directory)

#Set the theme, default,  dark,
pv.set_plot_theme("default")

rename_dict = {"DA_": "_DA_", "BVTV_": "_BVTV_", "__": "_"}


#Define the input name and read in to memory for visualization
mesh_name = "femur_consolidated_CtTh_scalars.vtk"
input_mesh = pv.read(mesh_name)

#For getting the p-value thresholds in the same format.
stats_dir = pathlib.Path(r"D:\Desktop\Canids\wxegSurf\Femur\CtTh")
os.chdir(stats_dir)
stats_mesh = pv.read("humerus_consolidated_CtTh_ttest.vtk")


#Output either a png or a pdf
get_scalar_screens(input_mesh=input_mesh,
                   scalars=["BVTV", "DA"],
                   limits=[0.0, 0.50],
                   consistent_limits=True,
                   n_of_bar_txt_portions=6,
                   output_type="pdf")

#If one scalar is being used or the range shoudl be somehting different
get_scalar_screens(input_mesh=input_mesh, scalars=["BVTV"], limits=[0.0, 0.60], consistent_limits=True, n_of_bar_txt_portions=6, output_type="pdf")
get_scalar_screens(input_mesh=input_mesh, scalars=["DA"], limits=[0.0, 0.50], consistent_limits=True, n_of_bar_txt_portions=6, output_type="pdf")
get_scalar_screens(input_mesh=input_mesh, scalars=["CtTh"], limits=[0.0, 2.50], consistent_limits=True, n_of_bar_txt_portions=6, output_type="pdf")


get_ttest_screens(input_mesh=stats_mesh,
                  scalars=["BVTV", "DA"],
                  estimate_limits=True,
                  n_of_bar_txt_portions=11,
                  output_type="pdf")

get_ttest_screens(input_mesh=stats_mesh, scalars=["BVTV"], estimate_limits=True, output_type="png")
get_ttest_screens(input_mesh=stats_mesh, scalars=["DA"], estimate_limits=True, output_type="png")
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
merge_pdfs(pdf_directory=stats_dir, string_match="", out_name="Canid_femur_CtTh_results.pdf", output_directory="")

dist_plot_dir = r"D:\Desktop\Canids\Results\Distplots"

merge_pdfs(pdf_directory=dist_plot_dir, string_match="", out_name="Canid_distplot_results.pdf", output_directory=dist_plot_dir)


#Bayes
directory = r"D:\Desktop\Sharon\visualize"
os.chdir(directory)

#Define the input name and read in to memory for visualization
mesh_name = "BSBV_smooth_bayes_HDI_range_mapped.vtk"

bayes_outputs = ["posterior_mean", "posterior_sd", "HDI_range", "posterior_ninetyseven", "posterior_three"]
for output in bayes_outputs:
    consolidate_vtk(input_mesh=mesh_name, out_name=f"radius_{output}", name_match=f"{output}",
                    bayes_match=True, scalars=["BVTV", "DA", "BSBV", "Tb_Sp", "Tb_Th"], pairwise=False)

mesh_name = "radius_posterior_mean.vtk"
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

stats_mesh = pv.read("radius_bayes_POS_thresholds.vtk")
rename_dict = {"_nonKWapes_": "_Homo_sapiens_", "_KWapes_": "_Pan_troglodytes_",
               "_OWM_": "_old_world_monkeys_", "__": "_", "_POS_": "_POS"}

stats_mesh = scalar_name_cleanup(input_mesh=stats_mesh, replace_dict=rename_dict)

get_bayes_thresh_screens(input_mesh=stats_mesh, scalars=["Tb_Sp", "Tb_Th"], limits=[0, 1],
                         estimate_limits=False, n_of_bar_txt_portions=11, output_type="pdf")

stats_mesh = pv.read("radius_bayes_CohenD_thresholds.vtk")
rename_dict = {"_nonKWapes_": "_Homo_sapiens_", "_KWapes_": "_Pan_troglodytes_",
               "_OWM_": "_old_world_monkeys_", "__": "_", "_POS_": "_POS"}

stats_mesh = scalar_name_cleanup(input_mesh=stats_mesh, replace_dict=rename_dict)

get_bayes_thresh_screens(input_mesh=stats_mesh, scalars=["BVTV", "DA", "BSBV", "Tb_Sp", "Tb_Th"], limits=False,
                         estimate_limits=True, n_of_bar_txt_portions=11, output_type="pdf")

merge_pdfs(pdf_directory=directory, string_match="", out_name="Radius_trabecular_stats_results.pdf", output_directory="")

directory = r"D:\Desktop\Sharon\visualize\trab_pdf"
os.chdir(directory)

merge_pdfs(pdf_directory=directory, string_match="", out_name="Radius_trabecular_results.pdf", output_directory="")
