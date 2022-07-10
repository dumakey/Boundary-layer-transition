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
from scipy.linalg import svdvals
from pydmd import DMD


def read_geometry(filepath):

    geodata = np.array(pd.read_csv(filepath,header=None))

    xLE = min(geodata[:,0])
    xTE = max(geodata[:,0])
    iLE = np.where(geodata[:,0] == xLE)[0][0]
    iTE = np.where(geodata[:,0] == xTE)[0][0]

    upper_data = geodata[iLE:iTE,:]
    lower_data = geodata[iTE+1:,:]

    return upper_data, lower_data

def read_forces_breakdown_file(filepath):

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

def generate_snapshot_grid_points(geodata, plane_segment_coords, grid_parameters, snapshot_output_dir):
    
    for (ID, geopath) in geodata.items():
        # Set parameters
        Nsegment = len(plane_segment_coords)  # choordwise number of segments
        Nz = grid_parameters['NL']  # number of points per snapshot (z direction)
        GR = grid_parameters['GR']  # growthrate
        if grid_parameters['wall_distance'] == 'constant':
            sum = np.sum(np.array([GR**(j-1) for j in range(1,Nz+1)]))
            delta0 = grid_parameters['deltaT']/sum
        else:
            delta0 = grid_parameters['delta0'] # snapshot first point z-coordinate

        if grid_parameters['DX'] != None:
            DX = grid_parameters['DX'] # choordwise snapshot spacing
            N = int(0.1/DX)  # number of snapshots per 0.1 * x/c snapshot
        else:
            N = grid_parameters['N1']
    
        # Read wall geometry
        upper_data, lower_data = read_geometry(geopath)
        x0 = upper_data[0,0]
        z0 = upper_data[0,2]
        c = max(upper_data[:,0]) - min(upper_data[:,0])
    
        # Create export directories
        if not os.path.exists(snapshot_output_dir):
            os.mkdir(snapshot_output_dir)
    
        US_DMD_snapshots_dir = os.path.join(snapshot_output_dir, 'US')
        if os.path.exists(US_DMD_snapshots_dir):
            rmtree(US_DMD_snapshots_dir)
        os.mkdir(US_DMD_snapshots_dir)
    
        LS_DMD_snapshots_dir = os.path.join(snapshot_output_dir, 'LS')
        if os.path.exists(LS_DMD_snapshots_dir):
            rmtree(LS_DMD_snapshots_dir)
        os.mkdir(LS_DMD_snapshots_dir)
    
        # Generate DMD snapshots
        y_snapshot_gpoints = upper_data[0,1] * np.ones((Nz))
        for i, x_c in enumerate(plane_segment_coords):
            print('Segment in chord: ' + str(x_c) + '\n')
            x_snapshot_gpoints = x0 + np.linspace(0,x_c*c,(i+1)*N)
            Nx = len(x_snapshot_gpoints)
    
            DZ = np.array([delta0 + delta0 * np.sum(np.array([GR**(j-1) for j in range(1,i+1)])) for i in range(1,Nz+1)])
    
            # UPPER SIDE SNAPSHOT GRID POINTS
            US_data_sampling = upper_data[upper_data[:,0] <= (i+2)*x_c*c,:]
            try:
                z0_US_snapshot_gpoints = interp1d(US_data_sampling[:,0], US_data_sampling[:,2],kind='quadratic')(x_snapshot_gpoints)
            except:
                z0_US_snapshot_gpoints = interp1d(US_data_sampling[:,0], US_data_sampling[:,2],fill_value='extrapolate')(x_snapshot_gpoints)
    
            US_snapshot_grid_points = np.zeros([Nz,2,Nx,(i+1)*N])
            for j in range(Nx):
                US_snapshot_grid_points[:,0,j,i] = x_snapshot_gpoints[j]
                US_snapshot_grid_points[:,1,j,i] = z0_US_snapshot_gpoints[j] + DZ
    
            # LOWER SIDE SNAPSHOT GRID POINTS
            LS_data_sampling = lower_data[lower_data[:,0] <= (i+2)*x_c*c,:]
            try:
                z0_LS_snapshot_gpoints = interp1d(LS_data_sampling[:,0], LS_data_sampling[:,2], kind='quadratic')(x_snapshot_gpoints)
            except:
                z0_LS_snapshot_gpoints = interp1d(LS_data_sampling[:,0], LS_data_sampling[:,2],fill_value='extrapolate')(x_snapshot_gpoints)
    
            LS_snapshot_grid_points = np.zeros([Nz,2,Nx,(i+1)*N])
            for j in range(Nx):
                LS_snapshot_grid_points[:,0,j,i] = x_snapshot_gpoints[j]
                LS_snapshot_grid_points[:,1,j,i] = z0_LS_snapshot_gpoints[j] - DZ
    
            # Export
            x_c_segment = ('%.2f' %x_c).split('.')
            DMD_dir_name = '0' + x_c_segment[0] + x_c_segment[1]
            US_DMD_snapshot_dir = os.path.join(US_DMD_snapshots_dir, DMD_dir_name)
            if not os.path.exists(US_DMD_snapshot_dir):
                os.mkdir(US_DMD_snapshot_dir)
    
            LS_DMD_snapshot_dir = os.path.join(LS_DMD_snapshots_dir, DMD_dir_name)
            if not os.path.exists(LS_DMD_snapshot_dir):
                os.mkdir(LS_DMD_snapshot_dir)
    
            for j in range(Nx):
                US_snapshot_coords = np.zeros((Nz,3))
                US_snapshot_coords[:,0] = US_snapshot_grid_points[:,0,j,i]
                US_snapshot_coords[:,1] = y_snapshot_gpoints
                US_snapshot_coords[:,2] = US_snapshot_grid_points[:,1,j,i]
                US_snapshot_df = pd.DataFrame(US_snapshot_coords, columns=['x','y','z'])
                csvname = os.path.join(US_DMD_snapshot_dir, 'US_xc=%s_snapshot_%d.csv' %(x_c,j+1))
                US_snapshot_df.to_csv(csvname, index=False, sep=',', decimal='.')
    
                LS_snapshot_coords = np.zeros((Nz,3))
                LS_snapshot_coords[:,0] = LS_snapshot_grid_points[:,0,j,i]
                LS_snapshot_coords[:,1] = y_snapshot_gpoints
                LS_snapshot_coords[:,2] = LS_snapshot_grid_points[:,1,j,i]
                LS_snapshot_df = pd.DataFrame(LS_snapshot_coords, columns=['x','y','z'])
                csvname = os.path.join(LS_DMD_snapshot_dir, 'LS_xc=%s_snapshot_%d.csv' %(x_c,j+1))
                LS_snapshot_df.to_csv(csvname, index=False, sep=',', decimal='.')
    
            '''
            # Plot
            fig, ax = plt.subplots(2)
            fig.suptitle('x/c = %.2f' % x_c)
            for j in range(Nx):
                ax[0].scatter(US_snapshot_grid_points[:,0,j,i], US_snapshot_grid_points[:,1,j,i])
                ax[1].scatter(LS_snapshot_grid_points[:,0,j,i], LS_snapshot_grid_points[:,1,j,i])
            ax[0].scatter(x_snapshot_gpoints, z0_US_snapshot_gpoints)
            #ax[0].axis(ymin=0,ymax=0.2*max(US_snapshot_grid_points[:,1,-1,i]))
            #ax[1].axis(ymin=0.2*min(LS_snapshot_grid_points[:,1,-1,i]),ymax=0)
            ax[1].scatter(x_snapshot_gpoints, z0_LS_snapshot_gpoints)
    
            ax[1].set_xlabel('x (m)')
            ax[0].set_ylabel('z (m)')
            ax[1].set_ylabel('z (m)')
            '''
            print()

def generate_snapshot_data(snapshots_input_data_dir, snapshots_output_data_dir, variables, dymform='ND',casepath=None):

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
    casedata_file = [os.path.join(casepath,'CFD',file) for file in os.listdir(os.path.join(casepath,'CFD'))
                     if file == 'forces_breakdown.dat'][0]
    casedata = read_forces_breakdown_file(casedata_file)
    if dymform == 'ND':
        if casedata['DIM_FORMULATION'] == 'DIM':
            Pr_nd = 1/casedata['REF']['Pref']
            rho_nd = 1/casedata['REF']['Rhoref']
            v_nd = 1/casedata['REF']['vref']
            T_nd = 1/casedata['REF']['Tref']
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

    sides = os.listdir(snapshots_input_data_dir)
    if os.path.exists(snapshots_output_data_dir):
        rmtree(snapshots_output_data_dir)
    for side in sides:
        print('Airfoil side: ' + side + '\n')

        snapshot_data_segments_dir = os.listdir(os.path.join(snapshots_input_data_dir,side))
        for xc_segment in snapshot_data_segments_dir:  # loop over each segment
            print('    Segment in chord: ' +xc_segment + '\n')

            # Read input snapshot data
            snapshot_input_data_dir = os.path.join(snapshots_input_data_dir, side, xc_segment)
            snapshot_input_data_files = [file for file in os.listdir(snapshot_input_data_dir) if file.endswith('.csv')]
            for snapshot_input_data_file in snapshot_input_data_files:  # loop over each snapshot defined in each segment
                # Structure output snapshot data
                snapshot_input_data = pd.read_csv(os.path.join(snapshot_input_data_dir,snapshot_input_data_file))
                snapshot_output_data = snapshot_input_data[mapped_variables].copy()

                if 'u' in variables:
                    snapshot_output_data['u'] = snapshot_input_data['Momentum_0']*v_nd/snapshot_input_data['Density']
                if 'v' in variables:
                    snapshot_output_data['v'] = snapshot_input_data['Momentum_1']*v_nd/snapshot_input_data['Density']
                if 'w' in variables:
                    snapshot_output_data['w'] = snapshot_input_data['Momentum_2']*v_nd/snapshot_input_data['Density']
                if 'P' in variables:
                    snapshot_output_data['Pressure'] = snapshot_input_data['Pressure'] * Pr_nd
                if 'rho' in variables:
                    snapshot_output_data['Density'] = snapshot_input_data['Density'] * rho_nd

                snapshot_output_data.columns = direct_variables + derived_variables

                # Export output snapshot data
                snapshot_output_data_dir = os.path.join(snapshots_output_data_dir, side, xc_segment)
                if not os.path.exists(snapshot_output_data_dir):
                    os.makedirs(snapshot_output_data_dir)

                i_snapshot = int(re.search('_snapshot_(\d+).*', snapshot_input_data_file).group(1))
                csvname = os.path.join(snapshot_output_data_dir, '%s_%s_DMD_snapshot_%d.csv' %(side,xc_segment,i_snapshot))
                snapshot_output_data.to_csv(csvname, index=False, sep=',', decimal='.')


def plot_eigs(eigs, snapshots_arrangement, eigs_ref, xc_segment, eig_type=None, computation=None, dx=1.0, export=False, export_dir=None):

    if computation == 'pydmd':
        c = 'b'
        title = 'PyDMD'
    elif computation == 'manual':
        c = 'r'
        title = 'manual'

    if eig_type == 'lambda':
        ylabel = '$\lambda_r$'
        xlabel = '$\lambda_i$'

        ylim = (-45,45)
        xlim = (0,800)
    elif eig_type == 'uc':
        ylabel = '$\mu_r$'
        xlabel = '$\mu_i$'

        ylim = (-1.25,1.5)
        xlim = (-1.25,1.5)

    fig, ax = plt.subplots()
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    j = 0
    for arrangement,color in snapshots_arrangement.items():
        mu_R = np.zeros_like(eigs[j])
        mu_I = np.zeros_like(eigs[j])
        for i in range(len(eigs[j])):
            mu_R[i] = np.real(eigs[j][i])
            mu_I[i] = np.imag(eigs[j][i])
        ax.scatter(mu_I, mu_R, color=color,label='x%s' %arrangement)
        j += 1
    if eig_type == 'uc':
        theta = np.linspace(0,2*np.pi,50)
        uc_x = np.cos(theta)
        uc_y = np.sin(theta)
        ax.plot(uc_x,uc_y,linestyle='--',color='k',linewidth=1)
    if xc_segment == '0020':
        ax.scatter(eigs_ref[:,0],eigs_ref[:,1],color='g',marker='o',label='Wu et al')

    ax.grid()
    plt.legend()
    fig.suptitle('%s %s eigenvalues, %s extraction' % (xc_segment,eig_type,title))
    if export:
        fig.savefig(os.path.join(export_dir, '%s_%s_eigenvalues_%s.png' %(xc_segment,eig_type,computation)), dpi=200)

def plot_BL_profile(snapshots, x, z, x_c, export=False, export_dir=None):


    if export:
        export_folder = os.path.join(export_dir,x_c)
        if os.path.exists(export_folder):
            rmtree(export_folder)
        os.makedirs(export_folder)

    plt.ioff()
    for i in range(snapshots.shape[1]):
        fig, ax = plt.subplots()
        ax.set_ylabel('$z\,$(m)')
        ax.set_xlabel('$u\,$')
        ax.plot(snapshots[:,i],z[:,i])
        ax.grid()
        fig.suptitle('$x/c\,=\,$%s\n$x\,=\,$%.4f' %(x_c,x[i]))
        if export:
            fig.savefig(os.path.join(export_folder, '%s_x=%.4f_blprofiles.png' %(x_c,x[i])), dpi=200)
            plt.close()

def filter_nan(X, *args):

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


def dmd_analysis(snapshots_data_dir, reference_data_dir, variables, snapshots_arrangement, export=False, export_dir=None):

    # Read data from reference case
    reference_data_files = [file for file in os.listdir(reference_data_dir) if file.endswith('.dat')]
    reference_data = {'circle':None, 'physical': None}
    for file in reference_data_files:
        if 'circle' in file:
            reference_data['circle'] = pd.read_csv(os.path.join(reference_data_dir,file),sep=' ').to_numpy()
        else:
            reference_data['physical'] = pd.read_csv(os.path.join(reference_data_dir,file),sep=' ').to_numpy()

    Nvar = len(variables)

    # SET EXPORT FOLDER
    if os.path.exists(export_dir):
        rmtree(export_dir)
    os.mkdir(export_dir)

    # Read snapshots
    sides = os.listdir(snapshots_data_dir)
    for side in sides:
        snapshot_data_segments_dir = os.listdir(os.path.join(snapshots_data_dir, side))
        dmd_eigs = [[]] *len(snapshot_data_segments_dir)

        for ix, xc_segment in enumerate(snapshot_data_segments_dir):  # loop over each segment
            print('Segment in chord: ' + xc_segment + '\n')
            # Read input snapshot data
            snapshot_data_files_sorting_idx = [int(file.split('snapshot_')[1].split('.csv')[0]) for file in
                                    os.listdir(os.path.join(snapshots_data_dir, side, xc_segment)) if file.endswith('.csv')]
            snapshot_data_files = [file for file in os.listdir(os.path.join(snapshots_data_dir, side, xc_segment))
                                               if file.endswith('.csv')]
            sorting_idx = np.argsort(snapshot_data_files_sorting_idx)
            snapshot_data_files = [snapshot_data_files[idx] for idx in sorting_idx]

            # Allocate dmd data container
            f = open(os.path.join(snapshots_data_dir,side,xc_segment,snapshot_data_files[0]),'r')
            reader = csv.reader(f, delimiter=',')
            Ndmd_x = len(snapshot_data_files)  # number of planes in chordwise direction
            Ndmd_z = len(list(reader)) - 1 # Number of points in normalwise direction
            snapshots = np.zeros([Ndmd_z*Nvar,Ndmd_x])
            x = np.zeros((Ndmd_x))
            for i, snapshot_data_file in enumerate(snapshot_data_files):  # loop over each snapshot defined in each segment
                # Structure output snapshot data
                snapshot_data = pd.read_csv(os.path.join(snapshots_data_dir, side, xc_segment, snapshot_data_file))
                x[i] = snapshot_data['x'][0]
                snapshots[:,i] = np.reshape(snapshot_data[variables].to_numpy().T,(Ndmd_z*Nvar,))

            # Determine snapshots x-spacing
            ds = x[1] - x[0]

            # FILTER SNAPSHOTS
            snapshots_filt = filter_nan(snapshots)
            Ndmd_x_filt = snapshots_filt.shape[1]
            #fig = plt.plot(svdvals(np.array([snapshot.flatten() for snapshot in snapshots_filt]).T), 'o')

            mu_til_uc_list = []
            mu_til_list = []
            W_til_list = []
            dmd_eigs_uc_list = []
            dmd_eigs_list = []
            for arr in snapshots_arrangement.keys():
                arrangement = np.arange(0,Ndmd_x_filt,int(arr),dtype=int)
                snapshots_filt_arranged = snapshots_filt[:,arrangement]

                # EIGENVALUE EXTRACTION PARAMETERS
                rf = int(0.5*len(arrangement))
                if rf > Ndmd_z:
                    rf = Ndmd_z

                # EIGENVALUE EXTRACTION
                # MANUAL PROCEDURE
                X = snapshots_filt_arranged[:,:-1]
                X2 = snapshots_filt_arranged[:,1:]
                U, s, Vh = svd(X)
                S = np.diag(s)
                Sinv = pinv(S)
                V = Vh.conj()
                Uh = U.conj()

                Atil = np.dot(Uh[:,0:rf].T, np.dot(np.dot(X2,V[:,0:rf]),Sinv[0:rf,0:rf]))
                mu_til_uc, W_til = np.linalg.eig(Atil)
                mu_til = np.log(mu_til_uc)/ds
                Phi = np.dot(np.dot(np.dot(X2,V[:,0:rf]),Sinv[0:rf,0:rf]),W_til)
                mu_til_uc_list.append(mu_til_uc)
                mu_til_list.append(mu_til)
                W_til_list.append(W_til)

                # compute time evolution
                b = np.dot(pinv(Phi),X[:,0])
                Psi = np.zeros([rf,Ndmd_x], dtype='complex')
                for i, _x in enumerate(x):
                    Psi[:,i] = np.multiply(np.power(mu_til_uc,_x/ds),b)

                '''
                y = np.linspace(max(snapshot_data['z']),min(snapshot_data['z']),Ndmd_z)
                fig, ax = plt.subplots(Nvar)
                for ivar in range(Nvar):
                    ax[ivar].plot(X[ivar*Ndmd_z:Ndmd_z*(ivar+1)],y)
                '''

                # PYDMD PROCEDURE
                dmd = DMD(svd_rank=rf, tlsq_rank=2, exact=True, opt=True)
                dmd.fit(snapshots_filt)
                dmd_eigs_uc = dmd.eigs
                dmd_eigs = np.log(dmd.eigs)/ds
                dmd_eigs_uc_list.append(dmd_eigs_uc)
                dmd_eigs_list.append(dmd_eigs)

            # PLOT
            plot_eigs(eigs=mu_til_uc_list,snapshots_arrangement=snapshots_arrangement,eigs_ref=reference_data['circle'],
                      xc_segment=xc_segment,eig_type='uc',computation='manual',export=True,export_dir=export_dir)
            plot_eigs(eigs=mu_til_list,snapshots_arrangement=snapshots_arrangement,eigs_ref=reference_data['physical'],
                      xc_segment=xc_segment,eig_type='lambda',computation='manual',dx=ds,export=True,export_dir=export_dir)

            # Plot
            plot_eigs(eigs=dmd_eigs_uc_list,snapshots_arrangement=snapshots_arrangement,eigs_ref=reference_data['circle']
                      ,xc_segment=xc_segment,eig_type='uc',computation='pydmd',export=True,export_dir=export_dir)
            plot_eigs(eigs=dmd_eigs_list,snapshots_arrangement=snapshots_arrangement,eigs_ref=reference_data['physical'],
                      xc_segment=xc_segment,eig_type='lambda',computation='pydmd',dx=ds,export=True,export_dir=export_dir)

            print()

def plot_BL_profiles(snapshots_data_dir, variables, export=False, export_dir=None):

    Nvar = len(variables)

    # SET EXPORT FOLDER
    if os.path.exists(export_dir):
        rmtree(export_dir)
    os.mkdir(export_dir)

    # Read snapshots
    sides = os.listdir(snapshots_data_dir)
    for side in sides:
        snapshot_data_segments_dir = os.listdir(os.path.join(snapshots_data_dir, side))

        for ix, xc_segment in enumerate(snapshot_data_segments_dir):  # loop over each segment
            print('Segment in chord: ' + xc_segment + '\n')
            # Read input snapshot data
            snapshot_data_files = [file for file in os.listdir(os.path.join(snapshots_data_dir, side, xc_segment))
                                         if file.endswith('.csv')]

            # Allocate dmd data container
            Ndmd_x = len(snapshot_data_files)  # number of planes in chordwise direction
            f = open(os.path.join(snapshots_data_dir,side,xc_segment,snapshot_data_files[0]),'r')
            reader = csv.reader(f, delimiter=',')
            Ndmd_z = len(list(reader)) - 1 # Number of points in normalwise direction
            snapshots = np.zeros([Ndmd_z*Nvar,Ndmd_x])
            x = np.zeros((Ndmd_x))
            z = np.zeros((Ndmd_z,Ndmd_x))
            for i, snapshot_data_file in enumerate(snapshot_data_files):  # loop over each snapshot defined in each segment
                # Structure output snapshot data
                snapshot_data = pd.read_csv(os.path.join(snapshots_data_dir, side, xc_segment, snapshot_data_file))
                x[i] = snapshot_data['x'][0]
                z[:,i] = snapshot_data['z'].to_numpy()
                snapshots[:,i] = np.reshape(snapshot_data[variables].to_numpy().T,(Ndmd_z*Nvar,))

            # FILTER SNAPSHOTS
            snapshots_filt, x_filt = filter_nan(snapshots,x)
            plot_BL_profile(snapshots_filt,x=x_filt,z=z,x_c=xc_segment,export=True,export_dir=export_dir)

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

# Generate some test data.
cases = {
    'NLF0416': r'E:\DMD\NLF0416\Geometry\geo\refined\NLF0416_y=0_refined.csv',
}
snapshot_coord_dir = r'C:\Users\juan.ramos\Altran\Proyectos\Transition\BL_transition\Cases\NLF0416_laminar\Postprocessing\snapshots_coords'
snapshot_input_dir = r'C:\Users\juan.ramos\Altran\Proyectos\Transition\BL_transition\Cases\NLF0416_laminar\Postprocessing\snapshots_bulkdata'
snapshot_output_dir = r'C:\Users\juan.ramos\Altran\Proyectos\Transition\BL_transition\Cases\NLF0416_laminar\Postprocessing\snapshots_structured'
blprofiles_output_dir = r'C:\Users\juan.ramos\Altran\Proyectos\Transition\BL_transition\Cases\NLF0416_laminar\Postprocessing\BL_profiles'
dmd_analysis_output_dir = r'C:\Users\juan.ramos\Altran\Proyectos\Transition\BL_transition\Cases\NLF0416_laminar\Postprocessing\DMD_analysis_output'
reference_case_dir = r'C:\Users\juan.ramos\Altran\Proyectos\Transition\BL_transition\Cases\NLF0416_laminar\Postprocessing\Reference_data'

snapshots_arrangement = {'1':'r','2':'b','3':'c'}
dmd_variables = ['u','w']
snapshot_variables = ['x', 'z'] + dmd_variables

grid_parameters = {
'wall_distance': 'delta0',
'deltaT': 0.01,
'delta0':8e-06,
'GR':1.1,
'NL': 40,
'DX': None,
'N1': 20}

generate_snapshot_grid_points(geodata=cases,plane_segment_coords=plane_segment_coords,grid_parameters=grid_parameters,snapshot_output_dir=snapshot_coord_dir)
#generate_snapshot_data(snapshots_input_data_dir=snapshot_input_dir,snapshots_output_data_dir=snapshot_output_dir,variables=snapshot_variables,dymform='ND',casepath=fbrdown_file)
#dmd_analysis(snapshots_data_dir=snapshot_output_dir,reference_data_dir=reference_case_dir,variables=dmd_variables,snapshots_arrangement=snapshots_arrangement,export=True,export_dir=dmd_analysis_output_dir)
#plot_BL_profiles(snapshots_data_dir=snapshot_output_dir,variables=dmd_variables,export=True,export_dir=blprofiles_output_dir)
print()