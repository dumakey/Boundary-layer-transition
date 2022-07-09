import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import csv
from math import asin, cos, sin, degrees, radians, sqrt
from scipy.interpolate import interp1d
from scipy.linalg import svd, pinv, svdvals
from shutil import rmtree
import scipy.integrate as integrate
from scipy.optimize import fsolve

def filter_nan(X, *args):
    '''
    Function to filter a array off NaN values
    :param X: (array) array to filter
    :param args: (array) additional array from which the same filtered columns from X are taken
    :return: (array) filtered array(s)
    '''
    n, m = X.shape

    # Column filtering
    X_container = np.zeros((n,m))
    ncol = 1
    if args:
        s = args[0]
        scoords_filt = []
        for i in range(m):
            if len(set(np.isnan(X[:,i]))) == 1:
                X_container[:,ncol-1] = X[:,i]
                scoords_filt.append(s[i])
                ncol += 1
        X_filt = X_container[:,0:ncol-1]

        scoords_filt_array = np.reshape(np.array(scoords_filt),(ncol-1,1))
        return X_filt, scoords_filt_array
    else:
        for i in range(m):
            if len(set(np.isnan(X[:,i]))) == 1:
                X_container[:,ncol-1] = X[:,i]
                ncol += 1
        X_filt = X_container[:,0:ncol-1]

        return X_filt

def read_geometry(filepath):
    '''
    Function to read the geometry (contours) of the airfoil to be analysed
    :param filepath: (str) directory path where the coordinates file is storaged
    :return: (arrays) coordinates of the airfoil contours, both upper and lower side
    '''
    geodata = pd.read_csv(filepath)

    xLE = geodata['Points_0'][0]
    xTE = max(geodata['Points_0'])
    iLE = np.where(geodata['Points_0'] == xLE)[0][0]
    iTE = np.where(geodata['Points_0'] == xTE)[0][0]

    # Read points coordinates
    xup = np.array(geodata['Points_0'][iLE:iTE])
    zup = np.array(geodata['Points_2'][iLE:iTE])
    xlow = np.array(geodata['Points_0'][iTE+1:])
    zlow = np.array(geodata['Points_2'][iTE+1:])
    y = np.array(geodata['Points_1'][0])

    # Read normals
    nxup = np.array(geodata['Normals_0'][iLE:iTE])
    nzup = np.array(geodata['Normals_2'][iLE:iTE])
    nxlow = np.array(geodata['Normals_0'][iTE+1:])
    nzlow = np.array(geodata['Normals_2'][iTE+1:])

    return xup, zup, xlow, zlow, y, nxup, nzup, nxlow, nzlow

def read_forces_breakdown_file(filepath):
    '''
    Function to scan the forces breakdown file (SU2 format) to gather data about the case to be analysed
    :param filepath: (str) directory path where the file is located
    :return: (dict) dictionary where all the relevant data is storaged
    '''

    print('Reading forces breakdown file...')
    with open(filepath,'r') as f:
        data = f.read()

    casedata ={'FS': dict.fromkeys(['AOA','Re','Mach','Pinf','Tinf','Rhoinf','muinf','Tt','Pt'],None),
               'REF': dict.fromkeys(['Pref','Tref','Rhoref','muref','vref','Sref','Lref'],None),
               'SOLVER': None,
               'DIM_FORMULATION': None,
               }

    ## FREESTREAM PARAMETERS ##
    match = re.search('Mach number:\s*(\d\.*\d*).', data)
    if match:
        casedata['FS']['Minf'] = float(match.group(1))

    match = re.search('Angle of attack \(AoA\):\s*(\d*)', data)
    if match:
        casedata['FS']['AOA'] = radians(float(match.group(1)))

    match = re.search('Reynolds number:\s*(\d+\.*\d+([e|E]\+*\d*)?).', data)
    if match:
        casedata['FS']['Re'] = float(match.group(1))

    match = re.search('Free-stream static pressure:\s*(\d+\.*\d*)', data)
    if match:
        casedata['FS']['Pinf'] = float(match.group(1))

    match = re.search('Free-stream temperature:\s*(\d+\.*\d*)', data)
    if match:
        casedata['FS']['Tinf'] = float(match.group(1))

    match = re.search('Free-stream density:\s*(\d+\.*\d*)', data)
    if match:
        casedata['FS']['Rhoinf'] = float(match.group(1))

    match = re.search('Free-stream viscosity:\s*(\d+\.*\d+([e|E]\-\d*)?)', data)
    if match:
        casedata['FS']['muinf'] = float(match.group(1))

    match = re.search('Free-stream total pressure:\s*(\d+\.*\d*)', data)
    if match:
        casedata['FS']['Pt'] = float(match.group(1))

    match = re.search('Free-stream total temperature:\s*(\d+\.*\d*)', data)
    if match:
        casedata['FS']['Tt'] = float(match.group(1))

    ## REFERENCE PARAMETERS ##
    match = re.search('The reference area is (\d+\.*\d*)', data)
    if match:
        casedata['REF']['Sref'] = float(match.group(1))

    match = re.search('The reference length is (\d+\.*\d*)', data)
    if match:
        casedata['REF']['Lref'] = float(match.group(1))

    match = re.search('Reference pressure:\s*(\d+\.*\d*)', data)
    if match:
        casedata['REF']['Pref'] = float(match.group(1))

    match = re.search('Reference temperature:\s*(\d+\.*\d*)', data)
    if match:
        casedata['REF']['Tref'] = float(match.group(1))

    match = re.search('Reference density:\s*(\d+\.*\d*)', data)
    if match:
        casedata['REF']['Rhoref'] = float(match.group(1))

    match = re.search('Reference viscosity:\s*(\d+\.*\d+([e|E]\-\d*)?)', data)
    if match:
        casedata['REF']['muref'] = float(match.group(1))

    match = re.search('Reference velocity:\s*(\d+\.*\d*)', data)
    if match:
        casedata['REF']['vref'] = float(match.group(1))

    match = re.search('Turbulence model:\s*(\w+)', data)
    if match:
        casedata['SOLVER'] = 'Viscous'

    match = re.search('Compressible Euler equations', data)
    if match:
        casedata['SOLVER'] = 'Inviscid'

    match = re.search('Non-Dimensional simulation \((.*)\s*at the farfield\)', data)
    if match:
        casedata['DIM_FORMULATION'] = 'NDIM'
    else:
        casedata['DIM_FORMULATION'] = 'DIM'
    '''        
        nondim_ref_magnitudes = match.group(1).replace(' ','').split(',')
        if 'P=1.0' in nondim_ref_magnitudes:
            casedata['DIM_FORMULATION'] = 'PRESS_EQ_ONE'
        elif 'V=1.0' in nondim_ref_magnitudes:
            casedata['DIM_FORMULATION'] = 'VEL_EQ_ONE'
        elif 'V=Mach' in nondim_ref_magnitudes:
            casedata['DIM_FORMULATION'] = 'VEL_EQ_MACH'
        '''
    return casedata

def plot_Re(Re, x, x_c, Re_crit, theta, export_dir):
    '''
    Function to plot the evolution of the Reynolds number along the chordwise direction, to determine the transition
    point at the location where Re = Re_crit
    :param Re: (array) Reynolds number along the chordwise direction
    :param x: (array) chordwise points
    :param x_c: (str) chord segment of analysis
    :param Re_crit: (float) critical Reynolds number
    :param theta: (str) BL thickness
    :param export_dir: (str) path directory where to storage plots
    '''
    export_folder = os.path.join(export_dir,x_c)
    if os.path.exists(export_folder):
        rmtree(export_folder)
    os.makedirs(export_folder)

    plt.ioff()
    fig, ax = plt.subplots()
    ax.set_xlabel('$x\,$(m)')
    ax.set_ylabel('$Re_{\\theta}\,$')
    ax.plot(x,Re,'b')
    ax.plot(x,Re_crit*np.ones_like(x),'r--')
    ax.grid()
    fig.suptitle('$x/c\,=\,$%s' %(x_c))
    fig.savefig(os.path.join(export_folder, 'Re_%s_profiles_%s.png' %(theta,x_c)), dpi=200)
    plt.close()

def plot_BL_velocity_component(V, x, z, x_c, magnitude, export_dir=None):
    '''
    Function to plot the evolution of one component of the velocity at the boundary layer region
    :param V: (array) component to be plotted
    :param x: (str) longitudinal (chordwise) position where to plot
    :param z: (array) segment along which to plot the V-component
    :param x_c: (str) chord segment
    :param magnitude: (str) component to be plotted
    :param export_dir: (str) directory path where to storage the plot
    '''

    export_folder = os.path.join(export_dir,x_c,magnitude)
    if os.path.exists(export_folder):
        rmtree(export_folder)
    os.makedirs(export_folder)

    plt.ioff()
    for i in range(V.shape[1]):
        fig, ax = plt.subplots()
        ax.set_ylabel('$z\,$(m)')
        ax.set_xlabel('$%s\,$' %magnitude)
        ax.plot(V[:,i],z[:,i],'b')
        ax.grid()
        fig.suptitle('$x/c\,=\,$%s\n$x\,=\,$%.4f' %(x_c,x[i]))
        fig.savefig(os.path.join(export_folder, '%s_x=%.4f_%s_blprofile.png' %(x_c,x[i],magnitude)), dpi=200)
        plt.close()

def plot_BL_velocity_profiles(casepath):
    '''
    Function to plot the evolution of the longitudinal (u) and transversal (w) components of the flow velocity along
    the normal direction of the wall
    :param casepath: (str) path to the folder where the solution files are storaged
    '''
    # SET EXPORT FOLDER
    export_folder = os.path.join(casepath, 'Postprocessing', 'BL_analysis', 'BL_profiles')
    if os.path.exists(export_folder):
        rmtree(export_folder)
    os.makedirs(export_folder)

    # Locate normals folder
    grid_normals_dir = os.path.join(casepath, 'Postprocessing', 'BL_analysis', 'snapshot_grid', 'grid_wall_normals')
    try:
        if os.path.exists(grid_normals_dir):
            pass
    except:
        print('There is no normals directory. Please generate grid coordinates and normals.')

    # Read snapshots
    snapshots_data_dir = os.path.join(casepath, 'Postprocessing', 'BL_analysis', 'snapshots_structured')
    sides = os.listdir(snapshots_data_dir)
    for side in sides:
        snapshot_segments = os.listdir(os.path.join(snapshots_data_dir, side))

        for _, x_c in enumerate(snapshot_segments):  # loop over each segment
            print('Segment in chord: ' + x_c + '\n')

            # Read normals
            normals_filepath = [os.path.join(grid_normals_dir, side, x_c, file) for file in
                            os.listdir(os.path.join(grid_normals_dir, side, x_c))][0]
            normals_data = pd.read_csv(normals_filepath)
            x = normals_data['x'].to_numpy()
            normals = np.array([normals_data['nx'], normals_data['nz']]).T

            # Read input snapshot data
            snapshot_data_files = [file for file in os.listdir(os.path.join(snapshots_data_dir, side, x_c))
                                   if file.endswith('.csv')]

            #snapshot_normals_file = [file for file in os.listdir(os.path.join(grid_normals_dir, side, x_c))
            #                         if file.endswith('.csv')]

            # Allocate dmd data container
            Ndmd_x = len(snapshot_data_files)  # number of planes in chordwise direction
            f = open(os.path.join(snapshots_data_dir, side, x_c, snapshot_data_files[0]), 'r')
            reader = csv.reader(f,delimiter=',')
            Ndmd_z = len(list(reader)) - 1  # Number of points in normalwise direction
            z = np.zeros((Ndmd_z,Ndmd_x))
            u = np.zeros((Ndmd_z,Ndmd_x))
            w = np.zeros((Ndmd_z,Ndmd_x))

            sorting_idx = [int(file.split('snapshot_')[1].split('.csv')[0]) for file in snapshot_data_files if
                           file.endswith('.csv')]
            sorted_idx = np.argsort(sorting_idx)
            data_files = [snapshot_data_files[idx] for idx in sorted_idx]

            for i, snapshot_data_file in enumerate(data_files):  # loop over each snapshot defined in each segment
                # Structure output snapshot data
                snapshot_data = pd.read_csv(os.path.join(snapshots_data_dir,side,x_c,snapshot_data_file))
                z[:,i] = snapshot_data['z'].to_numpy()

                U = np.reshape(snapshot_data['u'].to_numpy().T,(Ndmd_z,))
                W = np.reshape(snapshot_data['w'].to_numpy().T,(Ndmd_z,))

                sinalpha = -abs(normals[i,0])
                cosalpha = abs(normals[i,1])
                u[:,i] = U*cosalpha - W*sinalpha
                w[:,i] = U*sinalpha + W*cosalpha

            # tangent-component of velocity profile
            u_filt, x_filt = filter_nan(u,x)
            plot_BL_velocity_component(u_filt,x_filt,z,x_c,'u',export_folder)
            # normal-component of velocity profile
            w_filt, x_filt = filter_nan(w,x)
            plot_BL_velocity_component(w_filt,x_filt,z,x_c,'w',export_folder)

def plot_BL_scalar_profiles(casepath, variable):
    '''
    Function to plot the evolution of a specified variable along the normal direction of the wall
    :param casepath: (str) path to the folder where the solution files are storaged
    :param variable: (str) variable to plot
    '''
    # SET EXPORT FOLDER
    export_folder = os.path.join(casepath, 'Postprocessing', 'BL_analysis', 'BL_profiles')
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)

    # Read snapshots
    snapshots_data_dir = os.path.join(casepath, 'Postprocessing', 'BL_analysis', 'snapshots_structured')
    sides = os.listdir(snapshots_data_dir)
    for side in sides:
        snapshot_segments = os.listdir(os.path.join(snapshots_data_dir, side))

        for _, x_c in enumerate(snapshot_segments):  # loop over each segment
            print('Segment in chord: ' + x_c + '\n')

            # Read input snapshot data
            snapshot_data_files = [file for file in os.listdir(os.path.join(snapshots_data_dir, side, x_c))
                                   if file.endswith('.csv')]

            # Allocate dmd data container
            Ndmd_x = len(snapshot_data_files)  # number of planes in chordwise direction
            f = open(os.path.join(snapshots_data_dir, side, x_c, snapshot_data_files[0]), 'r')
            reader = csv.reader(f, delimiter=',')
            Ndmd_z = len(list(reader)) - 1  # Number of points in normalwise direction
            delta1 = np.zeros([Ndmd_x,])
            delta2 = np.zeros([Ndmd_x,])
            x = np.zeros((Ndmd_x,))
            z = np.zeros((Ndmd_z,Ndmd_x))
            p = np.zeros((Ndmd_z,Ndmd_x))

            sorting_idx = [int(file.split('snapshot_')[1].split('.csv')[0]) for file in snapshot_data_files if
                           file.endswith('.csv')]
            sorted_idx = np.argsort(sorting_idx)
            data_files = [snapshot_data_files[idx] for idx in sorted_idx]

            for i, snapshot_data_file in enumerate(data_files):  # loop over each snapshot defined in each segment
                # Structure output snapshot data
                snapshot_data = pd.read_csv(os.path.join(snapshots_data_dir,side,x_c,snapshot_data_file))
                x[i] = snapshot_data['x'].to_numpy()[0]
                z[:,i] = snapshot_data['z'].to_numpy()
                p[:,i] = np.reshape(snapshot_data[variable].to_numpy().T,(Ndmd_z,))

            p_filt, x_filt = filter_nan(p,x)
            plot_BL_velocity_component(p_filt,x_filt,z,x_c,variable,export_folder)

def generate_snapshot_grid(casepath, plane_segment_coords, grid_parameters):
    '''
    Function to generate the mesh of points where to interpolate the solution

    :param casepath: link to the folder where the case files are located
    :param plane_segment_coords: chordwise division into segments
    :param grid_parameters: boundary layer parameters

    '''
    # Set parameters
    Nsegment = len(plane_segment_coords)  # choordwise number of segments
    Nz = grid_parameters['NL']  # number of points per snapshot (z direction)
    GR = grid_parameters['GR']  # growthrate
    if grid_parameters['wall_distance'] == 'constant':
        sum = np.sum(np.array([GR ** (j - 1) for j in range(1, Nz + 1)]))
        delta0 = grid_parameters['deltaT'] / sum
    elif grid_parameters['wall_distance'] == 'delta0':
        delta0 = grid_parameters['delta0']  # snapshot first point z-coordinate

    if grid_parameters['DX'] != None:
        DX = grid_parameters['DX']  # choordwise snapshot spacing
        N = int(0.1 / DX)  # number of snapshots per 0.1 * x/c snapshot
    else:
        N = grid_parameters['N1']

    # Read wall geometry
    geofile = [os.path.join(casepath, 'Geometry', file) for file in os.listdir(os.path.join(casepath, 'Geometry')) if
               file.endswith('.csv')][0]
    xup, zup, xlow, zlow, yw, nxup, nzup, nxlow, nzlow = read_geometry(geofile)
    x0 = xup[0]
    z0 = zup[0]
    c = max(xup) - min(xup)

    # Create export directories
    export_data_folder = os.path.join(casepath, 'Postprocessing', 'BL_analysis')
    if not os.path.exists(export_data_folder):
        os.mkdir(export_data_folder)

    # Directory where to store upper side grid points
    grid_coords_export_dir = os.path.join(export_data_folder, 'snapshot_grid', 'grid_coords')
    if os.path.exists(grid_coords_export_dir):
        rmtree(grid_coords_export_dir)
    os.makedirs(grid_coords_export_dir)

    # Directory where to store upper side normals
    grid_normals_export_dir = os.path.join(export_data_folder, 'snapshot_grid', 'grid_wall_normals')
    if os.path.exists(grid_normals_export_dir):
        rmtree(grid_normals_export_dir)
    os.makedirs(grid_normals_export_dir)

    # Generate DMD snapshots
    for i, x_c in enumerate(plane_segment_coords):
        print('Segment in chord: ' + str(x_c) + '\n')

        # Define normal wall distance
        d = np.array(
            [delta0 + delta0 * np.sum(np.array([GR ** (j - 1) for j in range(1, i + 1)])) for i in range(1, Nz + 1)])

        # Compute x-coordinate wall points
        xwall = x0 + np.linspace(0, x_c * c, (i + 1) * N)
        Nx = len(xwall)

        # UPPER WALL POINTS
        idx = np.where(xup <= (i + 2) * x_c * c)
        x_upwall_sample = xup[idx]
        z_upwall_sample = zup[idx]
        nx_upwall_sample = nxup[idx]
        nz_upwall_sample = nzup[idx]

        # Compute z-coordinate wall points
        try:
            zwall_up = interp1d(x_upwall_sample, z_upwall_sample, kind='quadratic')(xwall)
            nx_upwall = interp1d(x_upwall_sample, nx_upwall_sample, kind='quadratic')(xwall)
            nz_upwall = interp1d(x_upwall_sample, nz_upwall_sample, kind='quadratic')(xwall)
        except:
            zwall_up = interp1d(x_upwall_sample, z_upwall_sample, fill_value='extrapolate')(xwall)
            nx_upwall = interp1d(x_upwall_sample, nx_upwall_sample, fill_value='extrapolate')(xwall)
            nz_upwall = interp1d(x_upwall_sample, nz_upwall_sample, fill_value='extrapolate')(xwall)

        grid_points_up = np.zeros([Nz, 2, Nx, (i + 1) * N])

        for j in range(Nx):
            lambda_x = nx_upwall[j] / np.sqrt(nx_upwall[j] ** 2 + nz_upwall[j] ** 2)
            lambda_z = nz_upwall[j] / np.sqrt(nx_upwall[j] ** 2 + nz_upwall[j] ** 2)

            grid_points_up[:, 0, j, i] = xwall[j] + lambda_x * d
            grid_points_up[:, 1, j, i] = zwall_up[j] + lambda_z * d

        # LOWER WALL POINTS
        idx = np.where(xlow <= (i + 2) * x_c * c)
        x_lowall_sample = xlow[idx]
        z_lowall_sample = zlow[idx]
        nx_lowall_sample = nxlow[idx]
        nz_lowall_sample = nzlow[idx]
        try:
            zwall_low = interp1d(x_lowall_sample, z_lowall_sample, kind='quadratic')(xwall)
            nx_lowall = interp1d(x_lowall_sample, nx_lowall_sample, kind='quadratic')(xwall)
            nz_lowall = interp1d(x_lowall_sample, nz_lowall_sample, kind='quadratic')(xwall)
        except:
            zwall_low = interp1d(x_lowall_sample, z_lowall_sample, fill_value='extrapolate')(xwall)
            nx_lowall = interp1d(x_lowall_sample, nx_lowall_sample, fill_value='extrapolate')(xwall)
            nz_lowall = interp1d(x_lowall_sample, nz_lowall_sample, fill_value='extrapolate')(xwall)

        grid_points_low = np.zeros([Nz, 2, Nx, (i + 1) * N])
        for j in range(Nx):
            lambda_x = nx_lowall[j] / np.sqrt(nx_lowall[j] ** 2 + nz_lowall[j] ** 2)
            lambda_z = nz_lowall[j] / np.sqrt(nx_lowall[j] ** 2 + nz_lowall[j] ** 2)

            grid_points_low[:, 0, j, i] = xwall[j] + lambda_x * d
            grid_points_low[:, 1, j, i] = zwall_low[j] + lambda_z * d

        # Export points coordinates
        x_c_segment = ('%.2f' % x_c).split('.')
        export_folder = '0' + x_c_segment[0] + x_c_segment[1]
        upper_grid_points_folder = os.path.join(grid_coords_export_dir, 'US', export_folder)
        os.makedirs(upper_grid_points_folder)

        lower_grid_points_folder = os.path.join(grid_coords_export_dir, 'LS', export_folder)
        os.makedirs(lower_grid_points_folder)

        for j in range(Nx):
            upper_snapshot_coords = np.zeros((Nz, 3))
            upper_snapshot_coords[:, 0] = grid_points_up[:, 0, j, i]
            upper_snapshot_coords[:, 1] = yw * np.ones((Nz,))
            upper_snapshot_coords[:, 2] = grid_points_up[:, 1, j, i]
            upper_snapshot_df = pd.DataFrame(upper_snapshot_coords, columns=['x', 'y', 'z'])
            csvname = os.path.join(upper_grid_points_folder, 'US_xc=%s_snapshot_%d.csv' % (x_c, j + 1))
            upper_snapshot_df.to_csv(csvname, index=False, sep=',', decimal='.')

            lower_snapshot_coords = np.zeros((Nz, 3))
            lower_snapshot_coords[:, 0] = grid_points_low[:, 0, j, i]
            lower_snapshot_coords[:, 1] = yw * np.ones((Nz,))
            lower_snapshot_coords[:, 2] = grid_points_low[:, 1, j, i]
            lower_snapshot_df = pd.DataFrame(lower_snapshot_coords, columns=['x', 'y', 'z'])
            csvname = os.path.join(lower_grid_points_folder, 'LS_xc=%s_snapshot_%d.csv' % (x_c, j + 1))
            lower_snapshot_df.to_csv(csvname, index=False, sep=',', decimal='.')

        # Export point normals coordinates
        upper_grid_normals_export_dir = os.path.join(grid_normals_export_dir, 'US', export_folder)
        os.makedirs(upper_grid_normals_export_dir)

        lower_grid_normals_export_dir = os.path.join(grid_normals_export_dir, 'LS', export_folder)
        os.makedirs(lower_grid_normals_export_dir)

        upper_snapshot_data = np.zeros((Nx, 6))
        upper_snapshot_data[:, 0] = xwall
        upper_snapshot_data[:, 1] = yw * np.ones((Nx,))
        upper_snapshot_data[:, 2] = zwall_up
        upper_snapshot_data[:, 3] = nx_upwall
        upper_snapshot_data[:, 4] = 0.0
        upper_snapshot_data[:, 5] = nz_upwall
        upper_snapshot_data_df = pd.DataFrame(upper_snapshot_data, columns=['x', 'y', 'z', 'nx', 'ny', 'nz'])
        csvname = os.path.join(upper_grid_normals_export_dir, 'US_xc=%s_snapshot_normals.csv' % (x_c))
        upper_snapshot_data_df.to_csv(csvname, index=False, sep=',', decimal='.')

        lower_snapshot_data = np.zeros((Nx, 6))
        lower_snapshot_data[:, 0] = xwall
        lower_snapshot_data[:, 1] = yw * np.ones((Nx,))
        lower_snapshot_data[:, 2] = zwall_low
        lower_snapshot_data[:, 3] = nx_lowall
        lower_snapshot_data[:, 4] = 0.0
        lower_snapshot_data[:, 5] = nz_lowall
        lower_snapshot_data_df = pd.DataFrame(lower_snapshot_data, columns=['x', 'y', 'z', 'nx', 'ny', 'nz'])
        csvname = os.path.join(lower_grid_normals_export_dir, 'LS_xc=%s_snapshot_normals.csv' % (x_c))
        lower_snapshot_data_df.to_csv(csvname, index=False, sep=',', decimal='.')

        '''
        # Plot
        fig, ax = plt.subplots(2)
        fig.suptitle('x/c = %.2f' % x_c)
        for j in range(Nx):
            ax[0].scatter(grid_points_up[:,0,j,i], grid_points_up[:,1,j,i])
            ax[1].scatter(grid_points_low[:,0,j,i], grid_points_low[:,1,j,i])
        ax[0].scatter(xwall,zwall_up)
        #ax[0].axis(ymin=0,ymax=0.2*max(grid_points_up[:,1,-1,i]))
        #ax[1].axis(ymin=0.2*min(grid_points_low[:,1,-1,i]),ymax=0)
        ax[1].scatter(xwall,zwall_low)

        ax[1].set_xlabel('x (m)')
        ax[0].set_ylabel('z (m)')
        ax[1].set_ylabel('z (m)')
        '''
        print()

def generate_snapshot_data(casepath, variables, dymform='ND'):
    '''
    Function to interpolate the solution at the grid of points generated with function "generate_grid_points"
    :param casepath: link to the folder where the case files are located
    :param variables: variables to interpolate
    :param dymform: string to specify whether to express the variables in their dimensional ('D') or dimensionless ('ND')
    form

    '''
    # Mapping between the name given in Paraview and the desired name for each snapshot variable
    variables_mapping = {
        'x': 'Points_0',
        'y': 'Points_1',
        'z': 'Points_2',
        'rhoU': 'Momentum_0',
        'rhoV': 'Momentum_1',
        'rhoW': 'Momentum_2',
        'M': 'Mach',
        'P': 'Pressure',
        'Cp': 'Pressure_Coefficient',
        'rho': 'Density',
        'T': 'Temperature',
    }

    mapped_variables = []
    # direct variables such as Pressure, Momentum components or Density
    direct_variables = [variable for variable in variables if variable in variables_mapping.keys()]
    # variables that are derived from Paraview variables, such as velocity components
    derived_variables = [variable for variable in variables if variable not in variables_mapping.keys()]
    for variable in direct_variables:
        mapped_variables.append(variables_mapping[variable])

    # Read basic information about the CFD case
    casedata_file = [os.path.join(casepath, 'CFD', file) for file in os.listdir(os.path.join(casepath, 'CFD'))
                     if file == 'forces_breakdown.dat'][0]
    casedata = read_forces_breakdown_file(casedata_file)
    if dymform == 'ND':
        if casedata['DIM_FORMULATION'] == 'DIM':
            Pr_nd = 1 / casedata['REF']['Pref']
            rho_nd = 1 / casedata['REF']['Rhoref']
            v_nd = 1 / casedata['REF']['vref']
            T_nd = 1 / casedata['REF']['Tref']
        else:
            Pr_nd = 1.0
            rho_nd = 1.0
            v_nd = 1.0
            T_nd = 1.0
    elif dymform == 'D':
        if casedata['DIM_FORMULATION'] == 'DIM':
            Pr_nd = 1.0
            rho_nd = 1.0
            v_nd = 1.0
            T_nd = 1.0
        else:
            Pr_nd = casedata['REF']['Pref']
            rho_nd = casedata['REF']['Rhoref']
            v_nd = casedata['REF']['vref']
            T_nd = casedata['REF']['Tref']

    input_data_dir = os.path.join(casepath, 'Postprocessing', 'BL_analysis', 'snapshots_bulkdata')
    sides = os.listdir(input_data_dir)
    output_data_dir = os.path.join(casepath, 'Postprocessing', 'BL_analysis', 'snapshots_structured')
    if os.path.exists(output_data_dir):
        rmtree(output_data_dir)
    for side in sides:
        print('Airfoil side: ' + side + '\n')

        snapshot_segments = os.listdir(os.path.join(input_data_dir, side))
        for x_c in snapshot_segments:  # loop over each segment
            print('    Segment in chord: ' + x_c + '\n')

            # Read input snapshot data
            snapshot_input_data_dir = os.path.join(input_data_dir, side, x_c)
            snapshot_input_data_files = [file for file in os.listdir(snapshot_input_data_dir) if file.endswith('.csv')]
            for snapshot_input_data_file in snapshot_input_data_files:  # loop over each snapshot defined in each segment
                # Structure output snapshot data
                snapshot_input_data = pd.read_csv(os.path.join(snapshot_input_data_dir, snapshot_input_data_file))
                snapshot_output_data = snapshot_input_data[mapped_variables].copy()

                if 'u' in variables:
                    snapshot_output_data['u'] = snapshot_input_data['Momentum_0'] * v_nd / snapshot_input_data[
                        'Density']
                if 'v' in variables:
                    snapshot_output_data['v'] = snapshot_input_data['Momentum_1'] * v_nd / snapshot_input_data[
                        'Density']
                if 'w' in variables:
                    snapshot_output_data['w'] = snapshot_input_data['Momentum_2'] * v_nd / snapshot_input_data[
                        'Density']
                if 'P' in variables:
                    snapshot_output_data['Pressure'] = snapshot_input_data['Pressure'] * Pr_nd
                if 'T' in variables:
                    snapshot_output_data['Temperature'] = snapshot_input_data['Temperature'] * T_nd
                if 'rho' in variables:
                    snapshot_output_data['Density'] = snapshot_input_data['Density'] * rho_nd

                snapshot_output_data.columns = direct_variables + derived_variables

                # Export output snapshot data
                snapshot_output_data_dir = os.path.join(output_data_dir, side, x_c)
                if not os.path.exists(snapshot_output_data_dir):
                    os.makedirs(snapshot_output_data_dir)

                i_snapshot = int(re.search('_snapshot_(\d+).*', snapshot_input_data_file).group(1))
                csvname = os.path.join(snapshot_output_data_dir, '%s_%s_DMD_snapshot_%d.csv' % (side, x_c, i_snapshot))
                snapshot_output_data.to_csv(csvname, index=False, sep=',', decimal='.')

def compute_BL_thickness(casepath, reference_data=False, plot=True):
    '''
    Function to compute the boundary layer thickness distribution (delta1, delta2)
    :param casepath: (str) link to the folder where the case files are located
    :param reference_data: (bool) boolean to take the Euler case and compute the speed distribucion at the wall (Ue)
    :param plot: (bool) boolean to specify whether to generate plots or not

    '''
    # SET EXPORT FOLDER
    export_folder = os.path.join(casepath,'Postprocessing','BL_analysis','BL_thickness')
    delta1_folder = os.path.join(export_folder,'delta1')
    delta2_folder = os.path.join(export_folder,'delta2')
    if os.path.exists(export_folder):
        rmtree(export_folder)
    os.makedirs(delta1_folder)
    os.makedirs(delta2_folder)

    # Locate normals folder
    grid_normals_dir = os.path.join(casepath,'Postprocessing','BL_analysis','snapshot_grid','grid_wall_normals')
    try:
        if os.path.exists(grid_normals_dir):
            pass
    except:
        print('There is no normals directory. Please generate grid coordinates and normals.')

    # Read snapshots
    snapshots_data_dir = os.path.join(casepath,'Postprocessing','BL_analysis','snapshots_structured')
    sides = os.listdir(snapshots_data_dir)
    for side in sides:
        print('Airfoil side: ' + side + '\n')
        snapshot_segments = os.listdir(os.path.join(snapshots_data_dir,side))

        # Create export folder
        os.mkdir(os.path.join(delta1_folder,side))
        os.mkdir(os.path.join(delta2_folder,side))

        for _, x_c in enumerate(snapshot_segments):  # loop over each segment
            print('Segment in chord: ' + x_c + '\n')

            # Read normals
            normals_filepath = [os.path.join(grid_normals_dir,side,x_c,file) for file in os.listdir(os.path.join(grid_normals_dir,side,x_c))][0]
            normals_data = pd.read_csv(normals_filepath)
            x = normals_data['x'].to_numpy()
            normals = np.array([normals_data['nx'],normals_data['nz']]).T

            # Read input snapshot data
            snapshot_data_files = [file for file in os.listdir(os.path.join(snapshots_data_dir, side, x_c))
                                         if file.endswith('.csv')]

            #snapshot_normals_file = [file for file in os.listdir(os.path.join(grid_normals_dir, side, x_c))
            #                             if file.endswith('.csv')]

            # Define allocation variables
            Ndmd_x = len(snapshot_data_files)  # number of planes in chordwise direction
            f = open(os.path.join(snapshots_data_dir,side,x_c,snapshot_data_files[0]),'r')
            reader = csv.reader(f, delimiter=',')
            Ndmd_z = len(list(reader)) - 1 # Number of points in normalwise direction

            # Allocate variables
            delta1 = np.zeros([Ndmd_x,])
            delta2 = np.zeros([Ndmd_x,])
            z = np.zeros((Ndmd_z,Ndmd_x))
            u = np.zeros((Ndmd_z,Ndmd_x))
            w = np.zeros((Ndmd_z,Ndmd_x))

            # Find indexes to sort arrays
            sorting_idx = [int(file.split('snapshot_')[1].split('.csv')[0]) for file in snapshot_data_files if file.endswith('.csv')]
            sorted_idx = np.argsort(sorting_idx)
            snapshot_data_files = [snapshot_data_files[idx] for idx in sorted_idx]

            for i, snapshot_data_file in enumerate(snapshot_data_files):  # loop over each snapshot defined in each segment
                # Structure output snapshot data
                snapshot_data = pd.read_csv(os.path.join(snapshots_data_dir,side,x_c,snapshot_data_file))
                z[:,i] = snapshot_data['z'].to_numpy()

                U = np.reshape(snapshot_data['u'].to_numpy().T,(Ndmd_z,))
                W = np.reshape(snapshot_data['w'].to_numpy().T,(Ndmd_z,))

                # Change of basis
                sinalpha = -abs(normals[i,0])
                cosalpha = abs(normals[i,1])
                u[:,i] = U*cosalpha - W*sinalpha
                w[:,i] = U*sinalpha + W*cosalpha

            # Boundary layer computation
            for i, snapshot_data_file in enumerate(snapshot_data_files):
                # Determine the upper limit of integration
                jmax = np.nanargmax(u[:,i])
                ue = u[jmax,i]
                delta_1_integrand = np.ones_like(u[:jmax,i]) - u[:jmax,i]/ue
                delta_2_integrand = u[:jmax,i]/ue*(np.ones_like(u[:jmax,i]) - u[:jmax,i]/ue)

                delta1[i] = integrate.trapz(delta_1_integrand,abs(z[:jmax,i]))
                delta2[i] = integrate.trapz(delta_2_integrand,abs(z[:jmax,i]))

            # Retrieve reference data with which to compare
            if reference_data == True:
                if side == 'US':
                    reference_data_dir = os.path.join(casepath,'Postprocessing','BL_analysis','Reference_data','BL_thickness',side)
                    ref_delta_files = [os.path.join(reference_data_dir,file) for file in os.listdir(reference_data_dir) if file.endswith('.csv')]

                    ref_delta1_file = [file for file in ref_delta_files if 'delta1' in file][0]
                    ref_delta2_file = [file for file in ref_delta_files if 'delta2' in file][0]

                    ref_delta1_df = pd.read_csv(ref_delta1_file)
                    ref_delta2_df = pd.read_csv(ref_delta2_file)

            delta1_df = pd.DataFrame(np.vstack((x,delta1)).T, columns=['x','delta1'])
            csvname = os.path.join(delta1_folder,side,'delta1_thickness_%s.csv' %x_c)
            delta1_df.to_csv(csvname,index=False,sep=',',decimal='.')

            delta2_df = pd.DataFrame(np.vstack((x,delta2)).T,columns=['x','delta2'])
            csvname = os.path.join(delta2_folder,side,'delta2_thickness_%s.csv' %x_c)
            delta2_df.to_csv(csvname,index=False,sep=',',decimal='.')

            if plot == True:
                fig, ax = plt.subplots(2)
                fig.suptitle('Boundary layer thickness distribution\nChordwise segment: %s, Airfoil side: %s' % (x_c,side))
                for j in range(2):
                    ax[0].plot(delta1_df.loc[:,'x'],delta1_df.loc[:,'delta1'],color='b',label='delta1')
                    if reference_data == True and side == 'US':
                        ax[0].plot(ref_delta1_df.loc[:,'x'],ref_delta1_df.loc[:,'delta1'],color='b',linestyle='--',
                                   label='delta1 TAU')
                    ax[0].grid()
                    ax[0].set_ylabel('delta (m)')

                    ax[1].plot(delta2_df.loc[:,'x'],delta2_df.loc[:,'delta2'],color='r',label='delta2')
                    if reference_data == True and side == 'US':
                        ax[1].plot(ref_delta2_df.loc[:,'x'],ref_delta2_df.loc[:,'delta2'],color='r',linestyle='--',
                                   label='delta2 TAU')
                    ax[1].grid()
                    ax[1].set_ylabel('delta (m)')
                    ax[1].set_xlabel('x/c')

                    print()
                plt.tight_layout()
                fig.savefig(os.path.join(export_folder, '%s_delta_thickness_%s.png' %(side,x_c)), dpi=200)

def compute_transition(casepath, freestream_conditions, Re_cr):
    '''
    Function to compute the transition onset of a case, given its freestream conditions as well as the critical Reynolds number
    :param casepath: (str) path to the solution files
    :param freestream_conditions: (dict) dictionary with the case's freestream conditions storaged
    :param Re_cr: (float) critical Reynolds number for each of the criteria (delta 1, delta 2). Dictionary structure
    '''
    # SET EXPORT FOLDER
    export_folder = os.path.join(casepath,'Postprocessing','BL_analysis','BL_transition')
    if os.path.exists(export_folder):
        rmtree(export_folder)
    os.makedirs(export_folder)

    snapshots_data_dir = os.path.join(casepath,'Postprocessing','BL_analysis','snapshots_structured')
    sides = os.listdir(snapshots_data_dir)
    for side in sides:
        print('Airfoil side: ' + side + '\n')
        delta_dir = os.path.join(casepath,'Postprocessing','BL_analysis','BL_thickness')
        delta1_files = [os.path.join(delta_dir,'delta1',side,file) for file in os.listdir(os.path.join(delta_dir,'delta1',side)) if file.endswith('.csv')]
        delta2_files = [os.path.join(delta_dir,'delta2',side,file) for file in os.listdir(os.path.join(delta_dir,'delta2',side)) if file.endswith('.csv')]

        euler_data_dir = os.path.join(casepath,'Postprocessing','BL_analysis','Euler_data')
        ue_files = [os.path.join(euler_data_dir,'Velocity_distribution',side,file) for file in
                    os.listdir(os.path.join(euler_data_dir,'Velocity_distribution',side))
                    if file.endswith('.csv')]
        rhoe_files = [os.path.join(euler_data_dir,'Density_distribution',side,file) for file in
                    os.listdir(os.path.join(euler_data_dir,'Density_distribution',side))
                    if file.endswith('.csv')]

        # Sort files according to x_c segment
        x_c_perc = [int(file.split(os.sep)[-1].split('.csv')[0].split('_')[-1][-2:]) for file in ue_files]
        sorted_idx = np.argsort(x_c_perc)
        ue_files = [ue_files[idx] for idx in sorted_idx]

        x_c_perc = [int(file.split(os.sep)[-1].split('.csv')[0].split('_')[-1][-2:]) for file in rhoe_files]
        sorted_idx = np.argsort(x_c_perc)
        rhoe_files = [rhoe_files[idx] for idx in sorted_idx]

        x_c_perc = [int(file.split(os.sep)[-1].split('.csv')[0].split('_')[-1][-2:]) for file in delta1_files]
        sorted_idx = np.argsort(x_c_perc)
        delta1_files = [delta1_files[idx] for idx in sorted_idx]

        x_c_perc = [int(file.split(os.sep)[-1].split('.csv')[0].split('_')[-1][-2:]) for file in delta2_files]
        sorted_idx = np.argsort(x_c_perc)
        delta2_files = [delta2_files[idx] for idx in sorted_idx]

        # Set export folder
        Re_theta1_folder = os.path.join(export_folder,'delta1',side)
        os.makedirs(Re_theta1_folder)
        Re_theta2_folder = os.path.join(export_folder,'delta2',side)
        os.makedirs(Re_theta2_folder)

        mu = freestream_conditions['mu']
        Nsegments = len(x_c_perc)
        x_c = [file.split(os.sep)[-1].split('.csv')[0].split('_')[-1] for file in delta2_files]
        for i in range(Nsegments):
            # Retrieve Ue component
            ue_data = pd.read_csv(ue_files[i])
            Ue = ue_data['Ue'].to_numpy()

            # Retrieve Ue component
            rhoe_data = pd.read_csv(rhoe_files[i])
            rhoe = rhoe_data['rhoe'].to_numpy()

            # Retrieve delta1
            delta1_data = pd.read_csv(delta1_files[i])
            x = delta1_data['x'].to_numpy()
            delta1 = delta1_data['delta1'].to_numpy()

            # Retrieve delta2
            delta2_data = pd.read_csv(delta2_files[i])
            x = delta2_data['x'].to_numpy()
            delta2 = delta2_data['delta2'].to_numpy()

            # Compute thickness Reynolds
            Re_theta1 = rhoe*Ue*delta1/mu
            Re_theta2 = rhoe*Ue*delta2/mu

            plot_Re(Re_theta1,x,x_c[i],Re_cr[side]['theta1'],'theta1',Re_theta1_folder)
            plot_Re(Re_theta2,x,x_c[i],Re_cr[side]['theta2'],'theta2',Re_theta2_folder)

        print()

def compute_Euler_freestream_conditions(casepath):
    '''
    Function to determine the freestream magnitudes evolution for the Euler case
    :param casepath: (str) path where the case files are storaged
    '''
    # SET EXPORT FOLDER
    export_folder = os.path.join(casepath,'Postprocessing','BL_analysis','Euler_data')

    # Locate Euler data folder
    euler_folder = os.path.join(casepath,'Postprocessing','BL_analysis','Euler_data','snapshots_structured')

    # Locate normals folder
    grid_normals_dir = os.path.join(casepath,'Postprocessing','BL_analysis','snapshot_grid','grid_wall_normals')
    try:
        if os.path.exists(grid_normals_dir):
            pass
    except:
        print('There is no normals directory. Please generate grid coordinates and normals.')

    # Read snapshots
    sides = os.listdir(euler_folder)
    for side in sides:
        # Set browsing folder
        structured_data_export_folder = os.listdir(os.path.join(euler_folder,side))
        # Set export folders
        Ue_folder = os.path.join(export_folder, 'Velocity_distribution',side)
        rhoe_folder = os.path.join(export_folder, 'Density_distribution',side)
        if not os.path.exists(Ue_folder):
            os.makedirs(Ue_folder)
        if not os.path.exists(rhoe_folder):
            os.makedirs(rhoe_folder)

        for _, x_c in enumerate(structured_data_export_folder):  # loop over each segment
            print('Segment in chord: ' + x_c + '\n')

            # Read normals
            normals_filepath = [os.path.join(grid_normals_dir,side,x_c,file) for file in os.listdir(os.path.join(grid_normals_dir,side,x_c))][0]
            normals_data = pd.read_csv(normals_filepath)
            x = normals_data['x'].to_numpy()
            normals = np.array([normals_data['nx'],normals_data['nz']]).T

            # Read input snapshot data
            snapshot_data_files = [file for file in os.listdir(os.path.join(euler_folder,side,x_c))
                                         if file.endswith('.csv')]

            snapshot_normals_file = [file for file in os.listdir(os.path.join(grid_normals_dir,side,x_c))
                                         if file.endswith('.csv')]

            # Allocate dmd data container
            Ndmd_x = len(snapshot_data_files)  # number of planes in chordwise direction
            ue = np.zeros((Ndmd_x,))
            rhoe = np.zeros((Ndmd_x,))

            sorting_idx = [int(file.split('snapshot_')[1].split('.csv')[0]) for file in snapshot_data_files if file.endswith('.csv')]
            sorted_idx = np.argsort(sorting_idx)
            data_files = [snapshot_data_files[idx] for idx in sorted_idx]

            for i, snapshot_data_file in enumerate(data_files):  # loop over each snapshot defined in each segment
                # Structure output snapshot data
                snapshot_euler_data = pd.read_csv(os.path.join(euler_folder,side,x_c,snapshot_data_file))
                Ue = snapshot_euler_data['u'][0]
                We = snapshot_euler_data['w'][0]

                sinalpha = -abs(normals[i,0])
                cosalpha = abs(normals[i,1])
                ue[i] = Ue*cosalpha - We*sinalpha

                rhoe[i] = snapshot_euler_data['rho'][0]

            ue_df = pd.DataFrame(np.vstack((x,ue)).T, columns=['x','Ue'])
            csvname = os.path.join(Ue_folder,'Ue_%s.csv' %x_c)
            ue_df.to_csv(csvname,index=False,sep=',',decimal='.')

            rhoe_df = pd.DataFrame(np.vstack((x,rhoe)).T, columns=['x','rhoe'])
            csvname = os.path.join(rhoe_folder,'rhoe_%s.csv' %x_c)
            rhoe_df.to_csv(csvname,index=False,sep=',',decimal='.')

        print()

#################################################### INPUTS ############################################################

cases = {
    'NLF0416_M03': r'C:\Users\juan.ramos\Altran\Proyectos\Transition\BL_transition\Cases\NLF0416_M03_A203',
}


plane_segment_coords = [
0.1,
0.2,
0.3,
0.4,
0.5,
0.6,
0.7,
0.8,
0.9,
]

analysis_variables = ['u','w','rho','P','T']
snapshot_variables = ['x', 'z'] + analysis_variables

grid_parameters = {
'wall_distance': 'delta0',
'deltaT': 0.01,
'delta0':8e-06,
'GR':1.1,
'NL': 40,
'DX': None, 
'N1': 20}

'''
# For Euler simulations
grid_parameters = {
'wall_distance': 'delta0',
'deltaT': 0.001,
'delta0':8e-06,
'GR':1.1,
'NL': 1,
'DX': None,
'N1': 20}
'''

Re = 4e6
rho = 1.225
U = 34
mu = rho*U/Re

freestream_conditions = {'mu':mu}
Re_cr = {
    'US': {
        'theta1':4400,
        'theta2':3200},
    'LS':{
        'theta1':5300,
        'theta2':4000,}
}

for (ID, casepath) in cases.items():
    #generate_snapshot_grid(casepath,plane_segment_coords,grid_parameters)
    #generate_snapshot_data(casepath,variables=snapshot_variables,dymform='D')
    #compute_BL_thickness(casepath,reference_data=False,plot=True)
    #compute_transition(casepath,freestream_conditions,Re_cr)
    plot_BL_velocity_profiles(casepath)
    #plot_BL_scalar_profiles(casepath,'T')
    #plot_BL_scalar_profiles(casepath,'P')
