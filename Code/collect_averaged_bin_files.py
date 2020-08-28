#Script to collect mesh registerd binary files in wxRegSurf v17 into one mean binary

# By Nick Stephens
# Sharon Kuo was there to break code and help out a lot. 
import os
import sys
import glob
import itertools
import pathlib
import struct
import shutil
import trimesh
import scipy.stats
import trimesh
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from timeit import default_timer as timer
from statsmodels.stats.anova import anova_lm
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter1d



def get_mean_case_2d(mean_fileList, outname, canonical_geo):
    np_array_list = []
    # Loop throug the file_list and read in the values
    for f in mean_fileList:
        # Use pandas to read the comma separated values
        data = pd.read_csv(f, sep=",")
        # Select the scalar column and push it to a numpy array
        data = data.iloc[31:,:].values
        print(len(data))
        # Append to the list defined above
        np_array_list.append(data)
        # Get the name from the loop
        name = f
        # Replace the "csv" string so it can be appended at the end
        name = name.replace("_CtTh.csv", "")
        # Print it out so we can see the progress
        print(name)
        # Use numpy save text to write out the indiviedual values
        np.savetxt(name + "_CtTh.csv", data, '%5.10f', delimiter=",")
    # Make sure we have a string for the output name
    outname = str(outname)
    # Get the individual means
    CtTh_results_list = mean_fileList
    list.sort(CtTh_results_list)
    CtTh_results_list
    print(len(CtTh_results_list))
    CtTh_results = pd.DataFrame()
    # Then use a for loop to append into the dataframe
    for items in CtTh_results_list:
        name = str(items)
        name = name.replace("canonical_", "")
        name = name.replace("_cortical_mapped_with_", "")
        name = name.replace("_dat.bin", "")
        name = name.replace(".csv", "")
        print(name)
        df = pd.read_csv(items)
        df = df.iloc[31:, :]
        print(df.shape)
        df.columns = [name]
        print(df.shape)
        CtTh_results = pd.concat([CtTh_results, df], axis=1)
    # Then we can see how many rows and the names of the columns
    CtTh_results.shape
    CtTh_results = CtTh_results.replace(0, np.NaN)
    CtTh_averages = CtTh_results.describe()
    CtTh_averages = CtTh_averages.T
    CtTh_averages.columns = ['node_count', 'CtTh_mean', 'CtTh_std', 'CtTh_min', '25%', '50%', '75%', 'CtTh_max']
    CtTh_averages.to_csv(outname + "_Ct_Th_stats.csv")

    # Stack the numpy arrays vertically (which changed from the last version)
    comb_np_array = np.hstack(np_array_list)
    print(comb_np_array)
    print("Shape of combined array:", comb_np_array.shape)
    # Convert this to a dataframe, and transpose it so the indvidual files are columnwise.
    CtTh_array = pd.DataFrame(comb_np_array)
    # Write out the matrix to a csv
    CtTh_array.to_csv(outname + "_all_CtTh_values.csv", index=False)
    # Create a new dataframe with pandas and get the mean across the rows
    CtTh_array = CtTh_array.replace(0, np.NaN)
    mean_CtTh = pd.DataFrame(CtTh_array.mean(axis=1, numeric_only=True))
    # Check to make sure the vertex values are the same
    print("number of vertex values:", mean_CtTh.shape[0])
    # Write out the mean to a csv
    mean_CtTh.to_csv(outname + "_mean_CtTh_2d.csv", index=False, sep=" ")  # pd.mean_bin
    # Write out scalar information
    # Define scalar name to match case output
    col_name = 'ESca1' + outName
    #Write out the case file with scalar per node to replace the one in the mean folder
    with open(outname + '_original_average.case', 'w', newline="") as fout:
        fout.write("FORMAT\ntype: ensight gold\n\nGEOMETRY\nmodel:                           ")
        fout.write(outname + "_original_average.geo\n\nVARIABLE\nscalar per node:  ")
        fout.write(col_name + "  " + "Esca1" + outname + "_original_average.Esca1")
        #nodes.to_csv(fout, header=False, index=False, sep= " ")
    with open('Esca1' + outname + '_original_average.Esca1', 'w', newline="") as fout:
        fout.write(col_name)
        fout.write('\npart\n         1\ntria3\n')
        mean_CtTh.to_csv(fout, header=False, index=False, sep= " ")

    #Calculate the standard deviation row wise and write out the case
    standard_dev = pd.DataFrame(CtTh_array.std(axis=1, numeric_only=True))
    print(standard_dev)
    with open('ESca1' + outname + '_original_standard_dev.Esca1', 'w', newline="") as fout:
        col_name2 = 'ESca1' + outname + "_original_standard_dev"
        fout.write(col_name2)
        fout.write('\npart\n         1\ntria3\n')
        standard_dev.to_csv(fout, header=False, index=False, sep=" ")
    with open(outname + "_original_standard_dev.case", 'w', newline="") as fout:
        fout.write('FORMAT\ntype: ensight gold\n\nGEOMETRY\nmodel:                           ')
        fout.write(outname + "_original_average.geo\n\nVARIABLE\nscalar per node:  ")
        fout.write(col_name2 + " " + "  " + col_name2 + '.Esca1')

    #Calculate the coefficient of variation and write the case
    coef_var = pd.DataFrame(CtTh_array.std(axis=1, numeric_only=True)/CtTh_array.mean(axis=1, numeric_only=True))
    with open('ESca1' + outname + '_original_coef_var.Esca1', 'w', newline="") as fout:
        col_name3 = 'ESca1' + outname + "_original_coef_var"
        fout.write(col_name3)
        fout.write('\npart\n         1\ntria3\n')
        coef_var.to_csv(fout, header=False, index=False, sep=" ")
    with open(outname + "_original_coef_var.case", 'w', newline="") as fout:
        fout.write('FORMAT\ntype: ensight gold\n\nGEOMETRY\nmodel:                           ')
        fout.write(outname + "_original_average.geo\n\nVARIABLE\nscalar per node:  ")
        fout.write(col_name3 + " " + "  " + col_name3 + '.Esca1')
    # Copy and rename the geometry from the original case file.
    # Note that this muyst be ascii, not binary as output by avizo
    shutil.copy(canonical_geo, outname + "_original_average.geo")

def read_thick_bin(name):
        name = str(name)
        thickness_binary = name + "_thick_dat.bin"
        f = open(thickness_binary, "rb")
        f.seek(256, os.SEEK_SET)
        data = np.fromfile(f,
                           dtype=np.float64,
                           sep="")
        data = pd.DataFrame(data)
        data.columns = ['CtTh']
        print(f"{name} has {data.shape[0]} scalar values in binary.")
        return data

def read_err_bin(name):
    name = str(name)
    error_binary = name + "_thick_err.bin"
    f = open(error_binary, "rb")
    f.seek(0, os.SEEK_SET)
    data = np.fromfile(f,
                       dtype=np.float64,
                       sep="")
    data = pd.DataFrame(data)
    print(f"{name} has {data.shape[0]} error estimates.")
    return data

def read_ply_bin(name):
    name = str(name)
    ply_file = name + ".ply"
    mesh_2d = trimesh.load(ply_file, process=False)
    points = pd.DataFrame(mesh_2d.vertices)
    points.columns = ['x', 'y', 'z']
    print(f"{name} has {points.shape[0]} vertices ply file.")
    return points

def read_smooth_thick_bin(name):
    name = str(name)
    thickness_binary = name + "_smthick_dat.bin"
    f = open(thickness_binary, "rb")
    f.seek(256, os.SEEK_SET)
    data = np.fromfile(f,
                       dtype=np.float64,
                       sep="")
    data = pd.DataFrame(data)
    data.columns = ['CtTh']
    print(f"{name} has {data.shape[0]} scalar values in binary.")
    return data


def get_cortical_point_cloud(name, smooth=True):
    name = str(name)
    name = name.replace(".ply", "")
    points = read_ply_bin(name)
    if smooth == True:
        CtTh_scalars = read_smooth_thick_bin(name)
    else:
        CtTh_scalars = read_thick_bin(name)
    points['CtTh'] = CtTh_scalars
    return points


def smooth_thickness_scalars(point_cloud, smooth_level=1):
    #error_thresh = float(error.mean() / (error.std() * 2))
    #scalar_thresh = float(scalars.mean() / (scalars.std() * 2))
    x_sorted = point_cloud.sort_values('x')
    y_sorted = point_cloud.sort_values('y')
    z_sorted = point_cloud.sort_values('z')
    x_smoothed = _smooth(x_sorted['CtTh'], smooth_level=smooth_level)
    y_smoothed = _smooth(y_sorted['CtTh'], smooth_level=smooth_level)
    z_smoothed = _smooth(z_sorted['CtTh'], smooth_level=smooth_level)
    x_sorted['CtTh'] = x_smoothed
    y_sorted['CtTh'] = y_smoothed
    z_sorted['CtTh'] = z_smoothed
    point_cloud['x_CtTh'] = x_sorted['CtTh']
    point_cloud['y_CtTh'] = y_sorted['CtTh']
    point_cloud['z_CtTh'] = z_sorted['CtTh']
    point_cloud['smoothed_CtTh'] = point_cloud.iloc[:, 4:].mean(axis=1)
    smoothed_points = point_cloud[['x', 'y', 'z', 'smoothed_CtTh']]

    return smoothed_points



def _smooth(scalars, smooth_level):
    smoothed = pd.DataFrame(gaussian_filter1d(scalars, smooth_level))
    return smoothed




def gather_binaries(outName, fileNames, canonical_geo):
    
    #Define the names for writing files
    outName = str(outName)
    #Join together the names for the output
    bin_output = outName + "_thick_dat.bin"
    bin_output_error = outName + "_thick_err.bin"

    #The string matching to pass to glob for the list
    fileNames = str(fileNames)
    bin_fileList = glob.glob("*" + fileNames + "*_dat.bin")
    print("Running analysis on {} files.".format(str(len(bin_fileList))))
    error_fileList = glob.glob("*" + fileNames + "*_err.bin")

    # Set up an empty list to stuff the array into
    np_array_list = []
    np_array_error = []

    #Loop through the bin and error fileLists and append the values to empty lists
    for f in bin_fileList:
        data = np.fromfile(f, dtype=np.float64, sep="")
        np_array_list.append(data)
        print("Points in binary: {}.".format(data.shape))
        name = str(f)
        name = name.replace("_dat.bin", "")
        #Save the individual files for later reference
        np.savetxt(name + "_CtTh.csv", data, '%5.10f', delimiter=",")
    for f in error_fileList:
        error = np.fromfile(f, dtype=np.float64, sep="")
        np_array_error.append(error)

    #Stack the values together and write them out
    binary_array = pd.DataFrame(np.vstack(np_array_list))
    binary_array.to_csv(outName + "_all_CtTh_values.csv", index=False)
    binary_array = binary_array.iloc[:, 32:].copy()
    print(binary_array)   
    comb_np_error = np.vstack(np_array_error)
    binary_error = pd.DataFrame(comb_np_error)
    binary_error.to_csv(outName + "_all_CtTh_reg_error_values.csv", index=False)
    print("There are {} points in each bin file.".format(binary_array.shape[1]))
    print("There are {} points in each error bin file.".format(binary_error.shape[1]))
    binary_error = binary_error.iloc[:,32:].copy()


    if binary_array.isnull().sum().sum() != 0:
        print("There are", binary_array.isnull().sum().sum(), "points without data in this set.")
        nan_rows = binary_array[binary_array.isnull().T.any().T]
        print("the mesh(s) associated with columns\n {} \n, need additional smoothing passes".format(str(nan_rows)))

    # Average across the thickness points across the columns with pandas
    else:
        binary_array = binary_array.replace(0, np.NaN)
        mean_bin = binary_array.mean(axis=0, numeric_only=True)
        print("number of vertex density values:", mean_bin.shape)
        mean_bin.to_csv(outName + ".csv", index=False)  # pd.mean_bin
        # Print the average registration error for the mean column, for reference
        mean_error = binary_error.mean(axis=0, numeric_only=True)
        print("Average registration error is:", mean_error.mean())

    # Write the BV/TV mean and registration error per point as individual double float binaries
    # WxRegsurf requires the final file name to end in _thickness.bin to see it.
    with open(bin_output, "wb") as f:
        binary = mean_bin.values
        print("Data type is:", binary.dtype)
        for b in binary:
            f.write(struct.pack('d', b))
        print("writing mean bin finished!")

    # Same with the err format
    with open(bin_output_error, "wb") as f:
        reg_error = mean_error.values
        print("Data type is:", binary.dtype)
        for b in reg_error:
            f.write(struct.pack('d', b))
        print("writing registration error bin finished!")

    # Write out the values to a csv
    mean_bin.to_csv(outName + "_node_values.csv", index=False, header=False)
    mean_error.to_csv(bin_output_error + "_node_values.csv", index=False, header=False)

    # Read them in and combine them to the scalar file
    canonical_geo = str(canonical_geo)
    mean_fileList = glob.glob("*" + fileNames + "*_CtTh.csv")

    get_mean_case_2d(mean_fileList, outName, canonical_geo)

def setup_models(data_files, group_list, group_indentifiers):
    """
    Places individual files into a properly formatted pandas dataframe.
    :param data_files:
    :param group_list:
    :param group_indentifiers:
    :return:
    """
    # This then gets zipped up into a dictionary to be passed on later
    dictionary = dict(zip(group_list, group_identifiers))
    print(dictionary)

    #Set up the first data file to merge with the others
    model_data_1 = pd.read_csv(data_files[0], sep= ',')
    model_data_1 = model_data_1.iloc[:,3:4]
    #setup the file_name as the column name
    file_name = str(data_files[0]).replace(".csv", "")
    model_data_1.columns = [str(file_name)]
    #Loop across the data file list, excluding the first,
    for file in data_files[1:]:
        file_name = file.replace(".csv", "")
        file_df = pd.read_csv(file, sep=',')
        file_df = file_df.iloc[:,3:4]
        file_df.columns = [str(file_name)]
        print(file_df.columns)
        model_data_1 = pd.merge(model_data_1, file_df, right_index=True, left_index=True)

    #Transpose the merged dataframe, reindex, and apply the header file_names
    new_model_data = model_data_1.T
    new_model_data['index'] = list(range(0, len(new_model_data)))
    new_model_data = new_model_data.reset_index().set_index('index')
    new_model_data.rename(columns={'level_0': 'Names'}, inplace=True)

    #Match the text in the index and create a new column

    #Set up an empty dataframe for the groups
    #If the groups and identifiers are the same length, you can just pass this to the pattern matching
    if len(dictionary.values()) == len(dictionary.keys()):

        #Print out the groups stored in the jeys
        print('\n', dictionary.keys())
        #Create the pattern to match by mapping the values separated by a pipe '|'
        pattern = '|'.join(map(str, dictionary.values()))
        print(pattern, '\n')

        #Create a temporary dataframe that is the size of the original
        df = pd.DataFrame(index=range(0, new_model_data.shape[0]))

        #Place the matched groups into this dataframe then insert it into the model data and reset the index
        df['group'] = pd.DataFrame(new_model_data['Names'].str.extract(r'(' + str(pattern) + ')'))
        new_model_data.insert(0, 'group', df['group'])
        new_model_data.sort_values("group", inplace=True)
        new_model_data.reset_index(drop=True, inplace=True)
    else:
        #If there are multiple identifiers we create a temporary dataframe then loop through the keys and their values
        new_df = pd.DataFrame()
        for keys in dictionary:
            print('\n', keys)
            pattern = '|'.join(map(str, dictionary[keys]))
            print(pattern, '\n')
            df = new_model_data[new_model_data['Names'].str.contains(r'(' + str(pattern) + ')')]
            df.insert(loc=0, column='group', value=str(keys))
            new_df = pd.concat([new_df, df])
            #df['group'] = pd.DataFrame(new_model_data['Names'].str.extract(r'(' + str(pattern) + ')'))
        new_model_data = new_df
        new_model_data.sort_values("group", inplace=True)
        new_model_data.reset_index(drop=True, inplace=True)
    return new_model_data

def mesh_info(mesh):
    '''
    A function to return basic information about a loaded 2d mesh
    :param mesh: A 2d mesh loaded through trimesh.load
    '''
    mesh_2d = mesh
    triangles = len(mesh_2d.triangles)
    points = len(mesh_2d.vertices)
    edgemin = mesh_2d.edges_unique_length.min()
    edgemax = mesh_2d.edges_unique_length.max()
    edgemean = mesh_2d.edges_unique_length.mean()
    if mesh_2d.is_volume == True:
        print("Mesh can be represented as a 3d volume.")
    else:
        if mesh_2d.is_watertight != True:
            print("Mesh has holes.")
        else:
            print("Mesh doesn't have holes, please check for other issues in Meshlab.")
    print("Mesh has {} faces and {} vertices.".format(triangles, points))
    print("Mesh edge length: \n           mean {:06.4f}, max {:06.4f}, min {:06.4f}".format(edgemean, edgemax, edgemin))

def get_mesh_points(mesh):
    mesh_info(mesh)
    points = pd.DataFrame(mesh_2d.vertices)
    points.columns
    points.columns = ["x", "y", "z"]
    return points

def get_fwhm(point_cloud_model):
    """
    Get an estimate of the FWHM from a point cloud model. Uses scipy stats to fit a Gaussian
    probability density funciton, which returns the mean and standard deviation (sigma)
    FWHM is then calcualted based on the standard deviation.
    FWHM = 2 * sqrt 2 log(2)  * sigma (https://en.wikipedia.org/wiki/Full_width_at_half_maximum)
    :param point_cloud_model:
    :return:
    """
    mean, std = scipy.stats.norm.fit(point_cloud_model)
    FWHM = 2 * np.sqrt(2 * np.log(2)) * std
    #FWHM = std * np.sqrt(8 * np.log(2))
    return FWHM


def get_resels_2d(mesh, fwhm):
    """
    Function modeified from https://github.com/puolival/multipy/blob/master/multipy/rft.py
    Used to estimate the number of resolution elements or resels from a 2d mesh.
    Input arguments:
    ================
    X : ndarray of floats
      Two-dimensional data array containing the analyzed data.
    fwhm : float
      Size of the smoothing kernel measured as full-width at half maximum
      (FWHM).
    """

    """Estimate the number of resolution elements. Here the idea is to simply
    compute how many FWHM sized blocsk are needed to cover the entire area of
    X. TODO: Replace with a better (proper?) method."""
    mesh_2d = trimesh.load(mesh_input_name)
    surface_area = mesh_2d.area    
    point_cloud_resels = float(surface_area * 2) / float(fwhm ** 2)
    return point_cloud_resels

def expected_ec_2d(resels, pvalue, degrees_free):
    """
    From https://github.com/puolival/multipy/blob/master/multipy/rft.py
    Function for computing the expected value of the Euler characteristic
    of a Gaussian field.
    Input arguments:
    ================
    R : float
      The number of resels or resolution elements.
    Z : float
      The applied Z-score threshold.
    """
    # From Worsley 1992
    pvalue = float(pvalue)
    degrees = int(degrees_free)
    z = np.asarray(np.linspace(0, 5, 1000000))
    EC = (resels * (4 * np.log(2)) * ((2 * np.pi) ** (-3. / 2)) * z) * np.exp((z ** 2) * (-0.5))
    pvalue_thresh = np.where(EC<pvalue)[0]
    pvalue_thresh = consecutive(pvalue_thresh)[1].min()
    z_thresh = z[pvalue_thresh]
    EC = 1 - scipy.stats.norm.cdf(z_thresh)

    # Worslsey 1996 Table II for t field
    EC_2d = resels * ((4 * np.log(2)) / ((2 * np.pi) ** (3 / 2))) * (scipy.special.gamma((degrees + 1) / 2)) / (
                ((degrees / 2) ** (1 / 2)) * scipy.special.gamma(degrees / 2)) * (1 + (z ** 2 / degrees)) ** (
                        (-1 / 2) * (degrees - 1)) * z
    pvalue_thresh = np.where(EC_2d < pvalue)[0]
    pvalue_thresh = consecutive(pvalue_thresh)[1].min()
    t_thresh = z[pvalue_thresh]
    EC_2d = 1 - scipy.stats.norm.cdf(t_thresh)
    return EC, z_thresh, EC_2d, t_thresh

def ttest_comparison(outname, frame, canonical_geo, uvalue, equal_var = True):
    """
    Function to do a number of pairwise two-tailed t-tests across a series out point clouds. Will write out a case file
    to be viewed in paraview, where the scalars are signed p-values thresholded using a provided uvalue. The uvalue
    should be calculated using the expected_ec_3d function provided.

    :param outname: The output name of the files.
    :param frame: The pandas with the point cloud scalars.
    :param canonical_geo: A .geo file representing the morphology of the point cloud.
    :param uvalue: The euler characteristic value obtained using the expected_ec_2d or expected_ec_3d function.
    :param equal_variance: If the variance of the samples isn't equal set to False to perform a Welch's test. Note
    that the obtain uvalue may not be appropriate for a Welch's t-test.
    :return: Returns a .case file and csv with values thresholded against euler characteristic values.
    """

    #Create an output name for the results
    outname = outname + "_ttest_results_"
    print(outname)

    #Assign the frame variable to the generic frame used
    frame = frame

    #Get the unique names in the group column and find all possible paired combinations
    group_list = frame['group'].unique().tolist()
    possibilities = list(itertools.combinations(group_list, 2))

    #Print out all the possible combinations
    print("Combinations:\n     {}".format('\n     '.join(' and '.join(map(str,sl)) for sl in possibilities)))


    #Loop through the possibilites and run a ttest for all of them.
    for p in possibilities:

        #Begin timing
        start = timer()

        #Set up a new dataframe and get the start and end of the compared groups
        isolated_df = frame[frame['group'].isin(p)]

        #Get the starting and ending for the paired groups
        start_1 = np.array(np.where(isolated_df["group"] == str(p[0]))).min()
        start_2 = np.array(np.where(isolated_df["group"] == str(p[1]))).min()
        end_1 = (1 + np.array(np.where(isolated_df["group"] == str(p[0]))).max())
        end_2 = (1 + np.array(np.where(isolated_df["group"] == str(p[1]))).max())

        #Print out the total individuals and the individuals per group for visual verification.
        print("\nN =", len(isolated_df))
        print("Running:\n      {} n={} vs {} n={} \n".format(
            str(p[0]), len(range(start_1, end_1)),
            str(p[1]), len(range(start_2, end_2))
        ))

        #Isolate the compared groups and put them in a new datafrane that will be written over on each loop.
        #This step performed a two-tailed t-test for each row of the dataframe.
        if equal_var == True:
            isolated_df = pd.DataFrame(scipy.stats.ttest_ind(isolated_df.iloc[int(start_1):int(end_1), 2:],
                                                             isolated_df.iloc[int(start_2):int(end_2), 2:],
                                                             axis=0, equal_var = True)).T
        else:
            isolated_df = pd.DataFrame(scipy.stats.ttest_ind(isolated_df.iloc[int(start_1):int(end_1), 2:],
                                                             isolated_df.iloc[int(start_2):int(end_2), 2:],
                                                             axis=0, equal_var = False)).T

        #Set the column names for the dataframe
        isolated_df.columns = ['T-score', 'p-value']
        print(isolated_df)

        #Create a new column in the dataframe for the absolute of the test statitic
        isolated_df['Absolute_T-score'] = abs(isolated_df['T-score'])

        #Threshold out the values and write 1.00 for the pvalues that fall below the defined uvalue
        isolated_df['thresh'] = np.where(isolated_df['T-score'] < float(uvalue),
                                         9999.00, #Assign a masking value
                                         isolated_df['T-score'])

        #Define the output name based on the outname and write out the dataframe
        new_name = str(outname) + str(p[0]) + "_vs_" + str(p[1])
        isolated_df.to_csv(new_name + ".csv", index=False, float_format='%.10f')

        #Isolate the scalar column amd pass it the write_case function
        scalars = isolated_df['thresh']
        print("\nWriting out {}.case file...".format(str(new_name)))
        write_case(outname=new_name, scalars=scalars, scalar_type="Tscore", canonical_geo=canonical_geo)
        end = timer()
        end_time = (end - start)
        print("Running tests took", end_time, "\n")

def get_degrees_free(group1_n, group2_n):
    """
    Get the degress of freedom for calculating the linear model corrections
    :param group1_n:
    :param group2_n:
    :return:
    """
    degrees_free = (int(group1_n) + int(group2_n) - 2)
    return degrees_free


def consecutive(data, stepsize=1):
    """
    Function to find consective elements in an array.
    :param data: numpy array.
    :param stepsize: how many values between elements before splitting the array.
    :return: Returns an array broken along the step size.
    """
    consecutive = np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
    return consecutive

############################################################
#                                                          #
#                   Begin operations                       #
#                                                          #
############################################################

#Define path to mean mesh registered binary files, along with input and output
dir = pathlib.Path(r'/mnt/ics/RyanLab/Projects/SKuo/Medtool/medtool_training/Point_cloud_cortical')
os.chdir(dir)

#dir = r(sys.argv[1])
#outName = sys.argv[2]

bone = "Radius_Dist"

ply_list = glob.glob("*.ply")
for f in ply_list:
    out_name = f.replace(".ply", "_cortical.csv")
    points =

#Read it in so we can get the resel value and uvalue for the pvalue thresholding
#point_cloud_model = pd.read_csv(point_cloud_model[0], header=None)
#canonical_geo = canonical_geo[0]

#Define the groups being analyzed
group_list = ["Pan_troglodytes", "Papio", "Tamandua", "Myrmecophaga_tridactyla"]
group1 = group_list[0]
group2 = group_list[1]
group3 = group_list[2]
group4 = group_list[3]

#They can have multiple identifiers for text matching or simply be a single list
#This can be species names "latrans, lupus" or group "black_earth"
#group1 = ["82970", "82971", "123027"]
#group2 = ["123028", "12303"]
#group3 = ["15200", "580"]

#If there are multiple texts comprisingls
# a group, put them into a list of lists
group_identifiers = [group1, group2, group3, group4]


################################################
# Shouldn't need to modify anything below here #
################################################

outName = "canonincal_" + str(bone) + "_original"
canonical_geo = glob.glob("*canonical*.geo")
canonical_geo = canonical_geo[0]

print(canonical_geo)

#gather_binaries(str(outName), "_mapped_with_*cortical_thick*", str(canonical_geo))

#groups = ["Pan_troglodytes", "Papio", "Tamandua", "Myrmecophaga_tridactyla"]

#for f in groups:
#    outName = str(f) + "_" + str(bone) + "_original"
#    matching = "_mapped_with_*" + str(f) + "*cortical_thick*"
#    gather_binaries(str(outName), str(matching), str(canonical_geo))


#Define the mesh file to import
mesh_input_name = glob.glob("*canonical*cortical.ply")
mesh_input_name = mesh_input_name[0]
print(mesh_input_name)

degrees_free = get_degrees_free(5, 6)

mesh_2d = trimesh.load(str(mesh_input_name))
points = get_mesh_points(mesh_2d)
fwhm = get_fwhm(points)
resel = get_resels_2d(mesh_2d, fwhm)
uvalue = expected_ec_2d(resel, 0.05, degrees_free=degrees_free)[3]
print(resel)
print(uvalue)


#Read in the various bvtv or da outputs to a lsit
data_files = glob.glob("*_CtTh_original_mapped.csv")
data_files.sort()
print(data_files)
print(len(data_files))

#Setup the dataframe for the ttests
model_data_ctth = setup_models(data_files, group_list, group_identifiers)
model_data_ctth.to_csv("Data_from_all_groups_CtTh.csv", index=False)

model_data_ctth = pd.read_csv("Data_from_all_groups_CtTh.csv")

ttest_comparison(outname="CtTh",
                 frame= model_data_ctth,
                 canonical_geo=canonical_geo,
                 uvalue=uvalue)

