"""
Python script to perform statistical comparisons on either point cloud models or a 2d mesh.


Author: NB Stephens
Date: 1st of March, 2019
"""
import os
import sys
import glob
import shutil
import pathlib
import trimesh
import platform
import itertools
import scipy.stats
import numpy as np
import pandas as pd
import pyvista as pv
import statsmodels.formula.api as smf
from scipy.spatial import distance
from timeit import default_timer as timer
from statsmodels.stats.anova import anova_lm


def get_mean_point_distance(point_cloud_model):
    """
    Depricated
    :param point_cloud_model:
    :return:
    """
    """
    :param point_cloud_model: 
    :return: 
    """

    # Get the point by point distance of the point cloud model
    distance_df = pd.DataFrame(
        distance.cdist(point_cloud_model, point_cloud_model, metric="euclidean")
    )
    mean_distance = distance_df.iloc[1:4, :].mean(axis=1).mean()
    return mean_distance


def get_fwhm(point_cloud_model):
    """
    Get an estimate of the FWHM from a point cloud model. Uses scipy stats to fit a Gaussian
    probability density function, which returns the mean and standard deviation (sigma)
    FWHM is then calculated based on the standard deviation.
    FWHM = 2 * sqrt 2 log(2)  * sigma (https://en.wikipedia.org/wiki/Full_width_at_half_maximum)
    :param point_cloud_model:
    :return:
    """
    mean, std = scipy.stats.norm.fit(point_cloud_model)
    FWHM = 2 * np.sqrt(2 * np.log(2)) * std
    # FWHM = std * np.sqrt(8 * np.log(2))
    return FWHM


def get_fwhm_rft1d(data1, data2):
    # From rft1d
    """
    Ported from rft1d package
    Estimate field smoothness (FWHM) from a set of random fields or a set of residuals.
    :param data1:
    :param data2:
    :return:
    """
    yA = data1
    yB = data2
    mA, mB = yA.mean(axis=0), yB.mean(axis=0)
    rA, rB = yA - mA, yB - mB
    residuals = np.vstack([rA, rB])

    ssq = (residuals ** 2).sum(axis=0)
    # ### gradient estimation (Method 1:  SPM5, SPM8)
    # dx     = np.diff(R, axis=1)
    # v      = (dx**2).sum(axis=0)
    # v      = np.append(v, 0)   #this is likely a mistake but is entered to match the code in SPM5 and SPM8;  including this yields a more conservative estimate of smoothness
    ### gradient estimation (Method 2)
    dy, dx = np.gradient(residuals)
    v = (dx ** 2).sum(axis=0)
    # normalize:
    eps = np.finfo(np.float).eps
    v /= ssq + eps
    # ### gradient estimation (Method 3)
    # dx     = np.diff(R, axis=1)
    # v      = (dx**2).sum(axis=0)
    # v     /= (ssq[:-1] + eps)
    # ignore zero-variance nodes:
    i = np.isnan(v)
    v = v[np.logical_not(i)]
    # global FWHM estimate:
    reselsPerNode = np.sqrt(v / (4 * np.log(2)))
    FWHM = 1 / reselsPerNode.mean()
    return FWHM


def get_resels_2d(point_cloud_model, mesh):
    """
    Used to estimate the number of resolution elements or resels from a 2d mesh.
    Function modified from https://github.com/puolival/multipy/blob/master/multipy/rft.py

    :param point_cloud_model: A point cloud generated from a surface mesh.
    :param mesh: The 2d mesh that was used to generate the points.
    :return: Returns the resolution elements estimate based on the surface area of the mesh.
    """
    """
    Estimate the number of resolution elements. Here the idea is to simply
    compute how many FWHM sized blocsk are needed to cover the entire area of the points.
    """

    mesh = pv.load(str(mesh))
    surface_area = mesh.area
    FWHM = get_fwhm(point_cloud_model)
    point_cloud_resels = float(surface_area) / float(FWHM ** 2)
    return point_cloud_resels


def get_resels_3d(point_cloud_model, mesh):  # , point_distance, ct_average_resolution):
    """
    A resel, from resolution element, is a concept used in image analysis. It describes the actual spatial
    image resolution in an image or a volume. The number of resels in the image will be lower or equal to
    the number of pixel/voxels in the image. In an actual image the resels can vary across the image and
    indeed the local resolution can be expressed as "resels per pixel" (or "resels per voxel"). In functional
    neuroimaging analysis, an estimate of the number of resels together with random field theory is used in
    statistical inference. Keith Worsley has proposed an estimate for the number of resels/roughness.
    Friston, Ashburner, Kiebel, Nichols, Penny (ed) 2006. pg 233 (Resels = Volume/FWHM**D)
    :param number_of_points:
    :param distance_between_points:
    :param average_resolution:
    :return:
    """
    mesh = pv.read(str(mesh))
    volume = mesh.volume
    FWHM = get_fwhm(point_cloud_model)
    point_cloud_resels = float(volume) / float(FWHM ** 3)
    return point_cloud_resels


def consecutive(data, stepsize=1):
    """
    Function to find consective elements in an array.
    :param data: numpy array.
    :param stepsize: how many values between elements before splitting the array.
    :return: Returns an array broken along the step size.
    """
    consecutive = np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)
    return consecutive


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
    z = np.asarray(np.linspace(0, 5, 100000))
    EC = (resels * (4 * np.log(2)) * ((2 * np.pi) ** (-3.0 / 2)) * z) * np.exp(
        (z ** 2) * (-0.5)
    )
    pvalue_thresh = np.where(EC < pvalue)[0]
    pvalue_thresh = consecutive(pvalue_thresh)[1].min()
    z_thresh = z[pvalue_thresh]
    EC = 1 - scipy.stats.norm.cdf(z_thresh)

    # Wolsey 1996 Table II for t field
    EC_2d = (
        resels
        * ((4 * np.log(2)) / ((2 * np.pi) ** (3 / 2)))
        * (scipy.special.gamma((degrees + 1) / 2))
        / (((degrees / 2) ** (1 / 2)) * scipy.special.gamma(degrees / 2))
        * (1 + (z ** 2 / degrees)) ** ((-1 / 2) * (degrees - 1))
        * z
    )
    pvalue_thresh = np.where(EC_2d < pvalue)[0]
    pvalue_thresh = consecutive(pvalue_thresh)[1].min()
    t_thresh = z[pvalue_thresh]
    EC_2d = 1 - scipy.stats.norm.cdf(t_thresh)
    return EC, z_thresh, EC_2d, t_thresh


'''
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
    degrees = int(degrees_free)


    z = np.asarray(np.linspace(0, 5, 10000))

    #EC = resels * (4*np.log(2)) * (2*np.pi)**(-3/2.) * z*np.exp((-1/2)*z**2)
    #Web example
    EC = resels * ((4*np.log(2)) * (2*np.pi)**(-3/2)) * z*np.exp(((-1 * z) * 2)/2)
    pvalue_thresh = np.where(EC < pvalue)
    pvalue_thresh = consecutive(pvalue_thresh[0])[0].max()
    EC = z[pvalue_thresh]

    #Wolsey 1996 Table II for t field
    EC_2d = resels * ((4 * np.log(2)) / ((2*np.pi) ** (3/2))) * (scipy.special.gamma((degrees + 1 )/2))/(((degrees/2)**(1/2)) * scipy.special.gamma(degrees/2)) * (1 + (z**2/degrees))**((-1/2)*(degrees-1))*z
    pvalue_thresh = np.where(EC_2d < pvalue)
    pvalue_thresh = consecutive(pvalue_thresh[0])[0].max()
    EC_2d = z[pvalue_thresh]


    return EC, EC_2d
'''


def z2p(z):
    """From z-score return p-value."""
    return 0.5 * (1 + scipy.special.erf(z / np.sqrt(2)))


def expected_ec_3d(resels, pvalue, degrees_free):
    """
    Function for the 3d Euler characteristic of a Gaussian field.
    Code adapted from Worlsey SurfStat and multipy rft(https://github.com/puolival/multipy/blob/master/multipy/rft.py).
    This will return a uvalue to threshold the results from a pairwise two-tailed t-test.
    Friston, Ashburner, Kiebel, Nichols, Penny (ed) 2006. pg 233
    :param Resels: Resels is the resilution element of the point cloud.
    :param pvalue: pvalue is the pvalue that you want to correct (i.e. 0.05)
    :return:
    """
    degrees = int(degrees_free)

    z = np.asarray(np.linspace(0, 20, 200000))

    # EC_3d = resels * (4* np.log(2)) * (2*np.pi)**(-3/2) * Zvalue*np.exp(-Zvalue*2/2)
    # EC = resels * ((4* np.log(2))**3/2) / ((2*np.pi)**2) * pvalue*np.exp(-1/2)**2
    EC = resels * (
        (((4 * np.log(2)) ** (3 / 2)) / ((2 * np.pi) ** 2))
        * (np.exp(((-1 * z) ** 2) / 2) * (z ** 2 - 1))
    )
    pvalue_thresh = np.where(EC < pvalue)[0]
    pvalue_thresh = consecutive(pvalue_thresh)[0].max()
    z_thresh = z[pvalue_thresh]
    EC = abs(1 - z2p(z_thresh))

    # Worlsey 1996 Table II, P3(t)
    # EC_1 = ((4*np.log(2))**(3/2))/((2*np.pi)**2)
    # EC_2 = ((1 + ((z**2) / degrees )) ** ((-1/2) * (degrees - 1)))
    # EC_3 = (((degrees - 1)/degrees) * ((z ** 2) - 1))
    EC_3d = (
        resels
        * (((4 * np.log(2)) ** (3 / 2)) / ((2 * np.pi) ** 2))
        * ((1 + ((z ** 2) / degrees)) ** ((-1 / 2) * (degrees - 1)))
        * (((degrees - 1) / degrees) * ((z ** 2) - 1))
    )
    pvalue_thresh = np.where(EC_3d < pvalue)[0]
    pvalue_thresh = consecutive(pvalue_thresh)[1].min()
    t_thresh = z[pvalue_thresh]
    EC_3d = abs(1 - z2p(t_thresh))

    # ((4 * np.log(2)) ** (3 / 2)) / ((2 * np.pi) ** 2) * (((degrees - 1)/degrees) * ((z ** 2) - 1)) * ((1 + ((z**2) / degrees )) ** ((-1/2) * (degrees - 1)))

    return EC, z_thresh, EC_3d, t_thresh


def get_degrees_free(group1_n, group2_n):
    """
    Get the degress of freedom for calculating the linear model corrections
    :param group1_n:
    :param group2_n:
    :return:
    """
    degrees_free = int(group1_n) + int(group2_n) - 2
    return degrees_free


def write_case(outname, scalars, scalar_type, canonical_geo):
    """
    Write out an .case file in Ensight Gold format for viewing in Paraview. Takes in a type of scalar and copies a
    previously written out .geo file for the geometry.

    :param outname: The output name of the case file.
    :param scalars: A single column from a pandas dataframe or series containign scalar values
    :param scalar_type: A string containing what the scalar is (e.g. BVTV, DA, p-values etc.)
    :param canonical_geo: A .geo file matching the morphology of the scalar infromation.
    :return: Writes out a .case file that can then be viewed in paraview.
    """
    scalar_type = str(scalar_type)
    outname = str(outname) + "_" + str(scalar_type)
    scalars = scalars
    canonical_geo = str(canonical_geo)

    # Open an Ensight Gold .case scalar text file to be written. The newline forces the windows newline type.
    with open("ESca1" + outname + ".Esca1", "w", newline="") as fout:
        # Define the column name so the files all reference one another
        column_name = "ESca1" + outname
        fout.write(column_name)

        # Write the first two lines, denoting a tetrahedral grid.
        fout.write("\npart\n         1\ntetra4\n")

        # The scalars are then written line by line using fout
        scalars.to_csv(fout, header=False, index=False, sep=" ", float_format="%.10f")
    # Open an Ensight Gold .case header
    with open(outname + ".case", "w", newline="") as fout:
        fout.write(
            "FORMAT\ntype: ensight gold\n\nGEOMETRY\nmodel:                           "
        )
        fout.write(outname + ".geo\n\nVARIABLE\nscalar per node:  ")
        fout.write(column_name + " " + "  " + column_name + ".Esca1")
        shutil.copy(canonical_geo, outname + ".geo")


# Function to loop over the columns in a dataframe and perform a linear model on the group
def linear_comparison(outname, frame, canonical_geo, uvalue):
    """
    Function to do a number of linear model comparisons across a series out point clouds. The comparison is for each
    point in the point cloud, where a model considering groups (full) is tested with one without groups (null/limited)
    to see if having groups provides a bettter explanation. This will write out a .case file to be viewed in paraview,
    where the scalars are p-values thresholded againt a provided uvalue. The uvalue should be calculated using the
    expected_ec_2d or expected_ec_3d function.

    :param outname: The output name of the files.
    :param frame: The pandas with the point cloud scalars.
    :param canonical_geo: A .geo file representing the morphology of the point cloud.
    :param uvalue: The euler characteristic value obtained using the expected_ec_2d or expected_ec_3d function.
    :return: Returns a .case file and csv with values thresholded against euler characteristic values.
    """

    # Create an output name for the results
    outname = outname + "_linear_results_"
    print(outname)
    # Assign the frame variable to the generic frame used
    frame = frame
    # Set up an empty dataframe to append to
    name = pd.DataFrame()
    # Loop over the columns in the dataframe
    results_df = pd.DataFrame()
    for column in frame.columns[2:]:
        # Begin timing
        start = timer()

        # Fit a full and reduced model
        full = smf.ols("frame[column] ~ group", data=frame).fit()
        reduced = smf.ols("frame[column] ~ 1", data=frame).fit()

        # Perform an anova on the two models
        anovaResults = anova_lm(reduced, full)

        # Subset the pvalues and fscores
        pvalue = anovaResults.T[1].iloc[5:6]
        fscore = anovaResults.T[1].iloc[4:5]
        degrees_free = anovaResults.T.iloc[0:1].values
        print(pvalue[0], fscore[0], degrees_free)

        # Append to the empty frame
        results_df = results_df.append([[float(pvalue), float(fscore)]])
    results_df.columns = ["p-value", "F-score"]
    print(results_df.head)
    results_df["thresh"] = np.where(
        results_df["p-value"] > float(uvalue), 1.00, results_df["p-value"]
    )
    results_df.to_csv("Linear_" + outname + ".csv", index=False, float_format="%.10f")
    scalars = results_df["thresh"]
    write_case(
        outname, scalars=scalars, scalar_type="pvalues", canonical_geo=canonical_geo
    )
    end = timer()
    end_time = end - start
    print("Running tests took", end_time, "\n")


def ttest_comparison(outname, frame, canonical_geo, resels, pvalue=0.05):
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

    # Create an output name for the results
    outname = f"{outname}_ttest_results_"
    print(outname)

    # Assign the frame variable to the generic frame used
    frame = frame

    # Get the unique names in the group column and find all possible paired combinations
    group_list = frame["group"].unique().tolist()
    possibilities = list(itertools.combinations(group_list, 2))

    # Print out all the possible combinations
    print(f"Pairwise comparisons:")
    [print(f'{" ":21}{x[0]} and {x[1]}') for x in possibilities]

    # Loop through the possibilites and run a ttest for all of them.
    for p in possibilities:

        # Set up a new dataframe and get the start and end of the compared groups
        isolated_df = frame[frame["group"].isin(p)]

        # Get the starting and ending for the paired groups
        start_1 = np.array(np.where(isolated_df["group"] == str(p[0]))).min()
        start_2 = np.array(np.where(isolated_df["group"] == str(p[1]))).min()
        end_1 = 1 + np.array(np.where(isolated_df["group"] == str(p[0]))).max()
        end_2 = 1 + np.array(np.where(isolated_df["group"] == str(p[1]))).max()

        group1_n = len(range(start_1, end_1))
        group2_n = len(range(start_2, end_2))

        degrees_free = get_degrees_free(group1_n=group1_n, group2_n=group2_n)
        # Print out the total individuals and the individuals per group for visual verification.
        print(f"\nN = {len(isolated_df)}")
        print(
            f"Running:\n{' ':8}{p[0]} n={group1_n} vs {p[1]} n={group2_n} with {degrees_free} degrees of freedom.\n"
        )

        try:
            # Begin timing
            start = timer()
            if pvalue != 0.05:
                pvalue = float(pvalue)
            uvalue = expected_ec_3d(resels, pvalue, degrees_free)
            print(uvalue)
            uvalue = uvalue[3]

            # Get the variance across the set to see if the set and then use non-equal if the ratio
            # of the highest value and lowest exceeds 1.5 treat as unequal variance
            # Following from http: // vassarstats.net / textbook / ch14pt1.html
            if type(isolated_df.iloc[0, 2:3][0]) == str:
                var1 = np.var(isolated_df.iloc[int(start_1) : int(end_1), 3:].values)
                var2 = np.var(isolated_df.iloc[int(start_2) : int(end_2), 3:].values)
            else:
                var1 = np.var(isolated_df.iloc[int(start_1) : int(end_1), 2:].values)
                var2 = np.var(isolated_df.iloc[int(start_2) : int(end_2), 2:].values)
            variances = [var1, var2]
            variances.sort(reverse=True)
            variance_ratio = variances[0] / variances[1]
            print(f"Variance ratio of {variance_ratio}...")

            # Isolate the compared groups and put them in a new datafrane that will be written over on each loop.
            # This step performed a two-tailed t-test for each row of the dataframe.
            if type(isolated_df.iloc[0, 2:3][0]) == str:
                if variance_ratio <= 1.5:
                    isolated_df = pd.DataFrame(
                        scipy.stats.ttest_ind(
                            isolated_df.iloc[int(start_1) : int(end_1), 3:],
                            isolated_df.iloc[int(start_2) : int(end_2), 3:],
                            axis=0,
                            equal_var=True,
                        )
                    ).T
                else:
                    isolated_df = pd.DataFrame(
                        scipy.stats.ttest_ind(
                            isolated_df.iloc[int(start_1) : int(end_1), 3:],
                            isolated_df.iloc[int(start_2) : int(end_2), 3:],
                            axis=0,
                            equal_var=False,
                        )
                    ).T

            else:
                if variance_ratio <= 1.5:
                    isolated_df = pd.DataFrame(
                        scipy.stats.ttest_ind(
                            isolated_df.iloc[int(start_1) : int(end_1), 2:],
                            isolated_df.iloc[int(start_2) : int(end_2), 2:],
                            axis=0,
                            equal_var=True,
                        )
                    ).T
                else:
                    isolated_df = pd.DataFrame(
                        scipy.stats.ttest_ind(
                            isolated_df.iloc[int(start_1) : int(end_1), 2:],
                            isolated_df.iloc[int(start_2) : int(end_2), 2:],
                            axis=0,
                            equal_var=False,
                        )
                    ).T
            # Set the column names for the dataframe
            isolated_df.columns = ["T-score", "p-value"]

            # Create a new column in the dataframe for the absolute of the test statitic
            isolated_df["Absolute_T-score"] = abs(isolated_df["T-score"])

            # Threshold out the values and write 1.00 for the pvalues that fall below the defined uvalue
            isolated_df["thresh"] = np.where(
                isolated_df["Absolute_T-score"] < float(uvalue),
                9999.00,  # Assign a masking value
                isolated_df["T-score"],
            )

            # Define the output name based on the outname and write out the dataframe
            new_name = f"{outname}{p[0]}_vs_{p[1]}"
            isolated_df.to_csv(new_name + ".csv", index=False, float_format="%.10f")

            # Isolate the scalar column amd pass it the write_case function
            scalars = isolated_df["thresh"]
            print(f"\nWriting out {new_name}.case file...")
            write_case(
                outname=new_name,
                scalars=scalars,
                scalar_type="Tscore",
                canonical_geo=canonical_geo,
            )
        except IndexError:
            print(
                f"Sample size is too small for {p[0]} vs {p[1]} to obtain a correction value!"
            )
        finally:
            end = timer()
            end_time = end - start
            print(f"Running tests took {end_time}\n")


def setup_models(data_files, group_list, group_identifiers):
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
    # Set up the first data file to merge with the others
    model_data_1 = pd.read_csv(data_files[0], sep=",", header=None)
    # model_data_1 = model_data_1.iloc[:,3:4]
    # setup the file_name as the column name
    file_name = str(data_files[0]).split("\\")[-1].replace(".csv", "")
    model_data_1.rename(columns={0: str(file_name)}, inplace=True)
    # Loop across the data file list, excluding the first,
    for file in data_files[1:]:
        file_name = file.split("\\")[-1]
        file_name = file_name.replace(".csv", "")
        file_df = pd.read_csv(file, sep=",", header=None)
        # file_df = file_df.iloc[:,3:4]
        file_df.rename(columns={0: str(file_name)}, inplace=True)
        print(file_df.columns)
        model_data_1 = pd.merge(
            model_data_1, file_df, right_index=True, left_index=True
        )

    # Transpose the merged dataframe, reindex, and apply the header file_names
    new_model_data = model_data_1.T
    new_model_data["index"] = list(range(0, len(new_model_data)))
    new_model_data = new_model_data.reset_index().set_index("index")
    new_model_data.rename(columns={"level_0": "Names"}, inplace=True)

    new_df = pd.DataFrame()
    for keys, values in dictionary.items():
        if type(values) == str:
            pattern = str(values)
            print(f"{pattern} \n")
        elif type(values) == list:
            pattern = "|".join(map(str, values))
            print(pattern, "\n")
        else:
            print(f"{keys} not understood.")
        df = new_model_data[
            new_model_data["Names"].str.contains(r"(" + str(pattern) + ")")
        ]
        df.insert(loc=0, column="group", value=str(keys))
        new_df = pd.concat([new_df, df])
        # df['group'] = pd.DataFrame(new_model_data['Names'].str.extract(r'(' + str(pattern) + ')'))
    new_model_data = new_df
    new_model_data.sort_values("group", inplace=True)
    new_model_data.reset_index(drop=True, inplace=True)
    return new_model_data


def get_necessary_files(matching_text="canonical"):
    # Grab the point cloud model, which will eb the first match in the list
    point_cloud_model = glob.glob(f"*{matching_text}*.csv")

    # Same for the canonical geo
    canonical_geo = glob.glob(f"*{matching_text}*.geo")

    # And for the ply
    canonical_ply = glob.glob(f"*{matching_text}*.ply")

    try:
        point_cloud_model = point_cloud_model[0]
    except IndexError:
        print(
            "Did not find canonical point cloud! Please make sure it is in the folder!"
        )
    finally:
        print(point_cloud_model)

    try:
        canonical_geo = canonical_geo[0]
    except IndexError:
        print("Did not find canonical geo cloud! Please make sure it is in the folder!")
    finally:
        print(canonical_geo)
    try:
        canonical_ply = canonical_ply[0]
    except IndexError:
        print("Did not find canonical ply! Please make sure it is in the folder!")
    finally:
        print(canonical_ply)
    return point_cloud_model, canonical_geo, canonical_ply


def get_data_files_for_scalars(
    group_list, group_identifiers, point_cloud_dir, scalar_list, max_normalixed=True
):
    for scale in scalar_list:
        print(f"Gathering results for {scale}...")
        scalar_dir = pathlib.Path(point_cloud_dir).joinpath(scale)
        if scalar_dir.exists():
            print(f"Found {scalar_dir}...")
            data_files = glob.glob(
                str(
                    pathlib.Path(
                        scalar_dir.joinpath(f"*{scale}*original_mapped_{scale}.csv")
                    )
                )
            )
            if max_normalixed == True:
                max_data_files = glob.glob(
                    str(
                        pathlib.Path(
                            scalar_dir.joinpath(
                                f"*{scale}*original_mapped_max_normalized_{scale}.csv"
                            )
                        )
                    )
                )
        else:
            print(f"Scalar dir not found, looking in root folder...")
            pathlib.Path.mkdir(scalar_dir)
            data_files = glob.glob(f"*{scale}*original_mapped_{scale}.csv")
            if max_normalixed == True:
                max_data_files = glob.glob(
                    f"*{scale}*original_mapped_max_normalized_{scale}.csv"
                )

        data_files.sort()
        print(data_files)
        print(len(data_files))

        # Setup the dataframe for the ttests
        model_data = setup_models(data_files, group_list, group_identifiers)
        print(model_data["group"])

        # Write all out
        fileOut = pathlib.Path(scalar_dir).joinpath(f"Data_from_all_groups_{scale}.csv")
        print(f"Writing out model data to {fileOut}")
        model_data.to_csv(str(fileOut), index=False)
        if max_normalixed == True:
            # Setup the dataframe for the ttests
            model_data = setup_models(max_data_files, group_list, group_identifiers)
            print(model_data["group"])

            # Write all out
            fileOut = pathlib.Path(scalar_dir).joinpath(
                f"Data_from_all_groups_max_normalized_{scale}.csv"
            )
            print(f"Writing out model data to {fileOut}")
            model_data.to_csv(str(fileOut), index=False)


def get_ttest_results_for_scalars(
    point_cloud_dir, scalar_list, canonical_geo, resels, pvalue, max_normalized=True
):
    for scale in scalar_list:
        print(f"Gathering results for {scale}...")
        scalar_dir = pathlib.Path(point_cloud_dir).joinpath(scale)
        model_file = pathlib.Path(scalar_dir).joinpath(
            f"Data_from_all_groups_{scale}.csv"
        )
        model_data = pd.read_csv(str(model_file))

        # This is hacky and needs to be replaced

        shutil.copy(canonical_geo, str(scalar_dir.joinpath(canonical_geo)))
        os.chdir(scalar_dir)

        # Do the pairwise T-tests for the group
        ttest_comparison(
            outname=str(scale),
            frame=model_data,
            canonical_geo=canonical_geo,
            resels=resels,
            pvalue=pvalue,
        )
        if max_normalized == True:
            model_file = pathlib.Path(scalar_dir).joinpath(
                f"Data_from_all_groups_max_normalized_{scale}.csv"
            )
            model_data = pd.read_csv(str(model_file))
            ttest_comparison(
                outname=f"{scale}_max_normalized",
                frame=model_data,
                canonical_geo=canonical_geo,
                resels=resels,
                pvalue=pvalue,
            )
    os.chdir(point_cloud_dir)


def get_data_files_for_vtk_scalars(
    group_list, group_identifiers, point_cloud_dir, max_normalixed=True
):
    results_dir = pathlib.Path(point_cloud_dir).joinpath("results")
    stats_dir = results_dir.joinpath("stats")
    if stats_dir.exists() == False:
        print(f"Creating stats directory {stats_dir}")
        pathlib.Path.mkdir(stats_dir)

    data_files = glob.glob(str(results_dir.joinpath(f"*_results.csv")))
    data_files.sort()
    print(f"Found {len(data_files)} scalar files...\n")

    for data in data_files:
        scalar_name = data.replace("_results.csv", "")
        if platform.system().lower() == "windows":
            scalar_name = scalar_name.split("\\")[-1]
        else:
            scalar_name = scalar_name.split("/")[-1]

        # Setup the dataframe for the ttests
        print(f"\nWriting out {scalar_name} model data...\n")
        model_data = format_for_models(
            data_files=data, group_list=group_list, group_identifiers=group_identifiers
        )
        model_data.to_csv(
            str(stats_dir.joinpath(f"Data_from_all_groups_{scalar_name}.csv")),
            sep=",",
            index=False,
        )

    if max_normalixed == True:
        data_files = glob.glob(
            str(results_dir.joinpath(f"*_results_max_normalized.csv"))
        )
        data_files.sort()
        print(f"Found {len(data_files)} max normalized scalar files...\n")

        for data in data_files:
            scalar_name = data.replace("_results_max_normalized.csv", "")
            if platform.system().lower() == "windows":
                scalar_name = scalar_name.split("\\")[-1]
            else:
                scalar_name = scalar_name.split("/")[-1]

            # Setup the dataframe for the ttests
            print(f"\nWriting out {scalar_name} model data...\n")
            model_data = format_for_models(
                data_files=data,
                group_list=group_list,
                group_identifiers=group_identifiers,
            )
            model_data.to_csv(
                str(
                    stats_dir.joinpath(
                        f"Data_from_all_groups_{scalar_name}_max_normalized.csv"
                    )
                ),
                sep=",",
                index=False,
            )


def format_for_models(data_files, group_list, group_identifiers):
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
    # Set up the first data file to merge with the others
    new_model_data = pd.read_csv(data_files, sep=",")

    # Transpose the merged dataframe, reindex, and apply the header file_names
    new_model_data = new_model_data.T
    new_model_data = new_model_data.iloc[1:]
    new_model_data["index"] = list(range(0, len(new_model_data)))
    new_model_data = new_model_data.reset_index().set_index("index")
    new_model_data.rename(columns={"level_0": "Names"}, inplace=True)

    new_df = pd.DataFrame()
    for keys, values in dictionary.items():
        if type(values) == str:
            pattern = str(values)
            print(f"{pattern} \n")
        elif type(values) == list:
            pattern = "|".join(map(str, values))
            print(pattern, "\n")
        else:
            print(f"{keys} not understood.")
        df = new_model_data[
            new_model_data["Names"].str.contains(r"(" + str(pattern) + ")")
        ]
        df.insert(loc=0, column="group", value=str(keys))
        new_df = pd.concat([new_df, df])
        # df['group'] = pd.DataFrame(new_model_data['Names'].str.extract(r'(' + str(pattern) + ')'))
    new_model_data = new_df
    new_model_data.sort_values("group", inplace=True)
    new_model_data.reset_index(drop=True, inplace=True)
    return new_model_data


def ttest_comparison_vtk(canonical_vtk, resels, point_cloud_dir, pvalue=0.05):

    stats_dir = pathlib.Path(point_cloud_dir).joinpath("results").joinpath("stats")
    mesh = pv.read(canonical_vtk)
    scalar_list = glob.glob(str(stats_dir.joinpath("*Data_from_all_groups_*")))

    for scalar in scalar_list:
        # Create an output name for the results
        scalar_name = scalar.replace("Data_from_all_groups_", "")
        scalar_name = scalar_name.replace(".csv", "")
        if platform.system().lower() == "windows":
            scalar_name = scalar_name.split("\\")[-1]
        else:
            scalar_name = scalar_name.split("/")[-1]
        outname = f"{scalar_name}_T_score_"
        stats_frame = pd.read_csv(scalar)
        # Get the unique names in the group column and find all possible paired combinations
        group_list = stats_frame["group"].unique().tolist()
        possibilities = list(itertools.combinations(group_list, 2))

        # Print out all the possible combinations
        print(f"Pairwise comparisons:")
        [print(f'{" ":21}{x[0]} and {x[1]}') for x in possibilities]

        # Loop through the possibilites and run a ttest for all of them.
        for p in possibilities:
            # Set up a new dataframe and get the start and end of the compared groups
            isolated_df = stats_frame[stats_frame["group"].isin(p)]
            # Get the starting and ending for the paired groups
            start_1 = np.array(np.where(isolated_df["group"] == str(p[0]))).min()
            start_2 = np.array(np.where(isolated_df["group"] == str(p[1]))).min()
            end_1 = 1 + np.array(np.where(isolated_df["group"] == str(p[0]))).max()
            end_2 = 1 + np.array(np.where(isolated_df["group"] == str(p[1]))).max()

            group1_n = len(range(start_1, end_1))
            group2_n = len(range(start_2, end_2))

            degrees_free = get_degrees_free(group1_n=group1_n, group2_n=group2_n)
            # Print out the total individuals and the individuals per group for visual verification.
            print(f"\nN = {len(isolated_df)}")
            print(
                f"Running:\n{' ':8}{p[0]} n={group1_n} vs {p[1]} n={group2_n} with {degrees_free} degrees of freedom.\n"
            )

            try:
                # Begin timing
                start = timer()
                if pvalue != 0.05:
                    pvalue = float(pvalue)
                uvalue = expected_ec_3d(resels, pvalue, degrees_free)
                print(uvalue)
                uvalue = uvalue[3]

                # Get the variance across the set to see if the set and then use non-equal if the ratio
                # of the highest value and lowest exceeds 1.5 treat as unequal variance
                # Following from http: // vassarstats.net / textbook / ch14pt1.html
                if type(isolated_df.iloc[0, 2:3][0]) == str:
                    var1 = np.var(
                        isolated_df.iloc[int(start_1) : int(end_1), 3:].values
                    )
                    var2 = np.var(
                        isolated_df.iloc[int(start_2) : int(end_2), 3:].values
                    )
                else:
                    var1 = np.var(
                        isolated_df.iloc[int(start_1) : int(end_1), 2:].values
                    )
                    var2 = np.var(
                        isolated_df.iloc[int(start_2) : int(end_2), 2:].values
                    )
                variances = [var1, var2]
                variances.sort(reverse=True)
                variance_ratio = variances[0] / variances[1]
                print(f"Variance ratio of {variance_ratio}...")

                # Isolate the compared groups and put them in a new datafrane that will be written over on each loop.
                # This step performed a two-tailed t-test for each row of the dataframe.
                if type(isolated_df.iloc[0, 2:3][0]) == str:
                    if variance_ratio <= 1.5:
                        isolated_df = pd.DataFrame(
                            scipy.stats.ttest_ind(
                                isolated_df.iloc[int(start_1) : int(end_1), 3:],
                                isolated_df.iloc[int(start_2) : int(end_2), 3:],
                                axis=0,
                                equal_var=True,
                            )
                        ).T
                    else:
                        isolated_df = pd.DataFrame(
                            scipy.stats.ttest_ind(
                                isolated_df.iloc[int(start_1) : int(end_1), 3:],
                                isolated_df.iloc[int(start_2) : int(end_2), 3:],
                                axis=0,
                                equal_var=False,
                            )
                        ).T
                else:
                    if variance_ratio <= 1.5:
                        isolated_df = pd.DataFrame(
                            scipy.stats.ttest_ind(
                                isolated_df.iloc[int(start_1) : int(end_1), 2:],
                                isolated_df.iloc[int(start_2) : int(end_2), 2:],
                                axis=0,
                                equal_var=True,
                            )
                        ).T
                    else:
                        isolated_df = pd.DataFrame(
                            scipy.stats.ttest_ind(
                                isolated_df.iloc[int(start_1) : int(end_1), 2:],
                                isolated_df.iloc[int(start_2) : int(end_2), 2:],
                                axis=0,
                                equal_var=False,
                            )
                        ).T

                # Set the column names for the dataframe
                isolated_df.columns = ["T-score", "p-value"]
                print(isolated_df)

                # Create a new column in the dataframe for the absolute of the test statitic
                isolated_df["Absolute_T-score"] = abs(isolated_df["T-score"])

                # Threshold out the values and write 1.00 for the pvalues that fall below the defined uvalue
                isolated_df["thresh"] = np.where(
                    isolated_df["Absolute_T-score"] < float(uvalue),
                    9999.00,  # Assign a masking value
                    isolated_df["T-score"],
                )

                # Define the output name based on the outname and write out the dataframe
                new_name = f"{outname}{p[0]}_vs_{p[1]}"
                isolated_df.to_csv(
                    str(stats_dir.joinpath(f"{new_name}.csv")),
                    index=False,
                    float_format="%.10f",
                )

                # Isolate the scalar column amd pass it the write_case function
                scalars = isolated_df["thresh"]
                mesh[f"{new_name}"] = scalars
            except IndexError:
                print(
                    f"Sample size is too small for {p[0]} vs {p[1]} to obtain a correction value!"
                )
            finally:
                end = timer()
                end_time = end - start
                print(f"Running tests took {end_time}\n")
    return mesh
