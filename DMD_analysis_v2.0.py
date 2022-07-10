import numpy as np
import matplotlib

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
from scipy.linalg import svdvals
from pydmd import DMD
matplotlib.use('Agg')

class DMD_scanner:
    '''
    Class to perform a DMD analysis: extraction of eigenvalues
    :param case: (dict) dictionary containing the paths of the case and reference cases to be analysed
    :param grid_parameters: (dict) parameters that define the DMD grid
    :param dmd_parameters: (dict) parameters relative to the DMD analysis
    '''
    def __init__(self, case, grid_parameters, dmd_parameters):
        self.casepath = case['casepath']
        self.reference_data_case = case['reference_case_path']
        self.dmd_variables = dmd_parameters['dmd_variables']
        self.snapshot_variables = dmd_parameters['snapshot_variables']
        self.snapshots_arrangement = dmd_parameters['snapshot_arrangement']
        self.analysis_segments = dmd_parameters['analysis_segments']
        self.grid_parameters = grid_parameters

    def filter_nan(self, X, *args):
        '''
        Method to filter a array off NaN values
        :param X: (array) array to filter
        :param args: (array) additional array from which the same filtered columns from X are taken
        :return: (array) filtered array(s)
        '''
        n, m = X.shape

        # Column filtering
        X_container = np.zeros((n, m))
        ncol = 1
        if args:
            s = args[0]
            scoords_filt = []
            for i in range(m):
                if len(set(np.isnan(X[:, i]))) == 1:
                    X_container[:, ncol - 1] = X[:, i]
                    scoords_filt.append(s[i])
                    ncol += 1
            X_filt = X_container[:, 0:ncol - 1]

            scoords_filt_array = np.reshape(np.array(scoords_filt), (ncol - 1, 1))
            return X_filt, scoords_filt_array
        else:
            for i in range(m):
                if len(set(np.isnan(X[:, i]))) == 1:
                    X_container[:, ncol - 1] = X[:, i]
                    ncol += 1
            X_filt = X_container[:, 0:ncol - 1]

            return X_filt

    def read_geometry(self, filepath):
        '''
        Method to read the geometry (contours) of the airfoil to be analysed
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

    def read_forces_breakdown_file(self, filepath):
        '''
        Method to scan the forces breakdown file (SU2 format) to gather data about the case to be analysed
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

        print('Forces breakdown file read.')

        return casedata

    def plot_eigs(self, eigs, eigs_ref, xc_segment, eig_type=None, computation=None, dx=1.0, export=False, export_dir=None):
        '''
        Method to plot the eigenvalues extracted by DMD methods
        :param eigs: (list) collection of the eigenvalues extracted at a chord segment
        :param eigs_ref: (array) collection of eigenvalues computed for a validation/reference case
        :param xc_segment: (str) chordwise direction segment of analysis
        :param eig_type: (str) type of eigenvalue: physical (lambda) or eigenvalue in the unit circle (mu)
        :param computation: (str) method of extraction
        :param dx: (str) snapshot spacing
        :param export: (bool) boolean to activate plot storage
        :param export_dir: (str) storage directory path
        '''

        if computation == 'pydmd':
            c = 'b'
            title = 'PyDMD'
        elif computation == 'manual':
            c = 'r'
            title = 'manual'

        fig, ax = plt.subplots()
        j = 0
        for arrangement,color in self.snapshots_arrangement.items():
            mu_R = np.zeros_like(eigs[j],dtype=float)
            mu_I = np.zeros_like(eigs[j],dtype=float)
            for i in range(len(eigs[j])):
                mu_R[i] = np.real(eigs[j][i])
                mu_I[i] = np.imag(eigs[j][i])

                if xc_segment != '0020':
                    if eig_type == 'lambda':
                        if 'ylim' in locals():
                            if mu_R[i] < ylim[0]:
                                ylim[0] = 1.1 * mu_R[i]
                            if mu_R[i] > ylim[1]:
                                ylim[1] = 1.1 * mu_R[i]
                        else:
                            ylim = [0.5*mu_R[i],0.5*mu_R[i]]

                        if 'xlim' in locals():
                            if mu_I[i] < xlim[0]:
                                xlim[0] = 1.1 * mu_I[i]
                            if mu_I[i] > xlim[1]:
                                xlim[1] = 1.1 * mu_I[i]
                        else:
                            xlim = [mu_I[i],mu_I[i]]
                else:
                    xlim = (-10,600)
                    ylim = (-50,20)

            ax.scatter(mu_I,mu_R,color=color,label='x%s' %arrangement)
            j += 1
        if eig_type == 'uc':
            theta = np.linspace(0,2*np.pi,50)
            uc_x = np.cos(theta)
            uc_y = np.sin(theta)
            ax.plot(uc_x,uc_y,linestyle='--',color='k',linewidth=1)
        if xc_segment == '0020':
            ax.scatter(eigs_ref[:,0],eigs_ref[:,1],color='g',marker='o',label='Wu et al')

        if eig_type == 'lambda':
            ylabel = '$\lambda_r$'
            xlabel = '$\lambda_i$'

            xlim = tuple(xlim)
            ylim = tuple(ylim)

            fig_title = 'Physical eigenvalues'

        elif eig_type == 'uc':
            ylabel = '$\mu_r$'
            xlabel = '$\mu_i$'
            ylim = (-1.1,1.1)
            xlim = (-1.1,1.1)

            fig_title = 'Eigenvalues in the unit circle'

        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.grid()
        plt.legend()
        fig.suptitle('%s, %s extraction\nSegment x/c: %s' % (fig_title,computation,xc_segment))
        if export:
            fig.savefig(os.path.join(export_dir, '%s_%s_eigenvalues_%s.png' %(xc_segment,eig_type,computation)), dpi=200)

    def generate_snapshot_grid(self):
        '''
        Method to generate the mesh of points where to interpolate the solution
        '''
        # Set parameters
        Nsegment = len(self.analysis_segments)  # choordwise number of segments
        Nz = self.grid_parameters['NL']  # number of points per snapshot (z direction)
        GR = self.grid_parameters['GR']  # growthrate
        if self.grid_parameters['wall_distance'] == 'constant':
            sum = np.sum(np.array([GR**(j-1) for j in range(1,Nz+1)]))
            delta0 = self.grid_parameters['deltaT']/sum
        elif self.grid_parameters['wall_distance'] == 'delta0':
            delta0 = self.grid_parameters['delta0'] # snapshot first point z-coordinate

        if self.grid_parameters['DX'] != None:
            DX = self.grid_parameters['DX'] # choordwise snapshot spacing
            N = int(0.1/DX)  # number of snapshots per 0.1 * x/c snapshot
        else:
            N = self.grid_parameters['N1']

        # Read wall geometry
        geofile = [os.path.join(self.casepath,'Geometry',file) for file in os.listdir(os.path.join(self.casepath, 'Geometry')) if file.endswith('.csv')][0]
        xup, zup, xlow, zlow, yw, nxup, nzup, nxlow, nzlow = self.read_geometry(geofile)
        x0 = xup[0]
        z0 = zup[0]
        c = max(xup) - min(xup)

        # Create export directories
        export_data_folder = os.path.join(self.casepath,'Postprocessing','BL_analysis')
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
        print('Beginning to generate the grid of analysis...')
        print('      Segment of analysis:')
        for i, x_c in enumerate(self.analysis_segments):
            print('      >> x/c: ' + str(x_c))

            # Define normal wall distance
            d = np.array([delta0 + delta0*np.sum(np.array([GR**(j-1) for j in range(1,i+1)])) for i in range(1,Nz+1)])

            # Compute x-coordinate wall points
            xwall = x0 + np.linspace(0,x_c*c,(i+1)*N)
            Nx = len(xwall)

            # UPPER WALL POINTS
            idx = np.where(xup <= (i+2)*x_c*c)
            x_upwall_sample = xup[idx]
            z_upwall_sample = zup[idx]
            nx_upwall_sample = nxup[idx]
            nz_upwall_sample = nzup[idx]

            # Compute z-coordinate wall points
            try:
                zwall_up = interp1d(x_upwall_sample,z_upwall_sample,kind='quadratic')(xwall)
                nx_upwall = interp1d(x_upwall_sample,nx_upwall_sample,kind='quadratic')(xwall)
                nz_upwall = interp1d(x_upwall_sample,nz_upwall_sample,kind='quadratic')(xwall)
            except:
                zwall_up = interp1d(x_upwall_sample,z_upwall_sample,fill_value='extrapolate')(xwall)
                nx_upwall = interp1d(x_upwall_sample,nx_upwall_sample,fill_value='extrapolate')(xwall)
                nz_upwall = interp1d(x_upwall_sample,nz_upwall_sample,fill_value='extrapolate')(xwall)

            grid_points_up = np.zeros([Nz,2,Nx,(i+1)*N])

            for j in range(Nx):
                lambda_x = nx_upwall[j]/np.sqrt(nx_upwall[j]**2 + nz_upwall[j]**2)
                lambda_z = nz_upwall[j]/np.sqrt(nx_upwall[j]**2 + nz_upwall[j]**2)

                grid_points_up[:,0,j,i] = xwall[j] + lambda_x*d
                grid_points_up[:,1,j,i] = zwall_up[j] + lambda_z*d

            # LOWER WALL POINTS
            idx = np.where(xlow <= (i+2)*x_c*c)
            x_lowall_sample = xlow[idx]
            z_lowall_sample = zlow[idx]
            nx_lowall_sample = nxlow[idx]
            nz_lowall_sample = nzlow[idx]
            try:
                zwall_low = interp1d(x_lowall_sample,z_lowall_sample,kind='quadratic')(xwall)
                nx_lowall = interp1d(x_lowall_sample,nx_lowall_sample,kind='quadratic')(xwall)
                nz_lowall = interp1d(x_lowall_sample,nz_lowall_sample,kind='quadratic')(xwall)
            except:
                zwall_low = interp1d(x_lowall_sample,z_lowall_sample,fill_value='extrapolate')(xwall)
                nx_lowall = interp1d(x_lowall_sample,nx_lowall_sample,fill_value='extrapolate')(xwall)
                nz_lowall = interp1d(x_lowall_sample,nz_lowall_sample,fill_value='extrapolate')(xwall)

            grid_points_low = np.zeros([Nz,2,Nx,(i+1)*N])
            for j in range(Nx):
                lambda_x = nx_lowall[j]/np.sqrt(nx_lowall[j]**2 + nz_lowall[j]**2)
                lambda_z = nz_lowall[j]/np.sqrt(nx_lowall[j]**2 + nz_lowall[j]**2)

                grid_points_low[:,0,j,i] = xwall[j] + lambda_x*d
                grid_points_low[:,1,j,i] = zwall_low[j] + lambda_z*d

            # Export points coordinates
            x_c_segment = ('%.2f' %x_c).split('.')
            export_folder = '0' + x_c_segment[0] + x_c_segment[1]
            upper_grid_points_folder = os.path.join(grid_coords_export_dir, 'US', export_folder)
            os.makedirs(upper_grid_points_folder)

            lower_grid_points_folder = os.path.join(grid_coords_export_dir, 'LS', export_folder)
            os.makedirs(lower_grid_points_folder)

            for j in range(Nx):
                upper_snapshot_coords = np.zeros((Nz,3))
                upper_snapshot_coords[:,0] = grid_points_up[:,0,j,i]
                upper_snapshot_coords[:,1] = yw * np.ones((Nz,))
                upper_snapshot_coords[:,2] = grid_points_up[:,1,j,i]
                upper_snapshot_df = pd.DataFrame(upper_snapshot_coords, columns=['x','y','z'])
                csvname = os.path.join(upper_grid_points_folder, 'US_xc=%s_snapshot_%d.csv' %(x_c,j+1))
                upper_snapshot_df.to_csv(csvname, index=False, sep=',', decimal='.')

                lower_snapshot_coords = np.zeros((Nz,3))
                lower_snapshot_coords[:,0] = grid_points_low[:,0,j,i]
                lower_snapshot_coords[:,1] = yw * np.ones((Nz,))
                lower_snapshot_coords[:,2] = grid_points_low[:,1,j,i]
                lower_snapshot_df = pd.DataFrame(lower_snapshot_coords, columns=['x','y','z'])
                csvname = os.path.join(lower_grid_points_folder, 'LS_xc=%s_snapshot_%d.csv' %(x_c,j+1))
                lower_snapshot_df.to_csv(csvname, index=False, sep=',', decimal='.')

            # Export point normals coordinates
            upper_grid_normals_export_dir = os.path.join(grid_normals_export_dir, 'US', export_folder)
            os.makedirs(upper_grid_normals_export_dir)

            lower_grid_normals_export_dir = os.path.join(grid_normals_export_dir, 'LS', export_folder)
            os.makedirs(lower_grid_normals_export_dir)

            upper_snapshot_data = np.zeros((Nx,6))
            upper_snapshot_data[:,0] = xwall
            upper_snapshot_data[:,1] = yw * np.ones((Nx,))
            upper_snapshot_data[:,2] = zwall_up
            upper_snapshot_data[:,3] = nx_upwall
            upper_snapshot_data[:,4] = 0.0
            upper_snapshot_data[:,5] = nz_upwall
            upper_snapshot_data_df = pd.DataFrame(upper_snapshot_data, columns=['x','y','z','nx','ny','nz'])
            csvname = os.path.join(upper_grid_normals_export_dir, 'US_xc=%s_snapshot_normals.csv' %(x_c))
            upper_snapshot_data_df.to_csv(csvname, index=False, sep=',', decimal='.')

            lower_snapshot_data = np.zeros((Nx,6))
            lower_snapshot_data[:,0] = xwall
            lower_snapshot_data[:,1] = yw * np.ones((Nx,))
            lower_snapshot_data[:,2] = zwall_low
            lower_snapshot_data[:,3] = nx_lowall
            lower_snapshot_data[:,4] = 0.0
            lower_snapshot_data[:,5] = nz_lowall
            lower_snapshot_data_df = pd.DataFrame(lower_snapshot_data, columns=['x','y','z','nx','ny','nz'])
            csvname = os.path.join(lower_grid_normals_export_dir, 'LS_xc=%s_snapshot_normals.csv' %(x_c))
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

        print('Grid generated.')

    def generate_snapshot_data(self, dymform='ND'):
        '''
        Method to interpolate the solution at the grid of points generated with function "generate_grid_points"

        :param dymform: (str) parameter to specify whether to express the variables in their dimensional ('D') or dimensionless ('ND')
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
        direct_variables = [variable for variable in self.snapshot_variables if variable in variables_mapping.keys()]
        # variables that are derived from Paraview variables, such as velocity components
        derived_variables = [variable for variable in self.snapshot_variables if variable not in variables_mapping.keys()]
        for variable in direct_variables:
            mapped_variables.append(variables_mapping[variable])

        # Read basic information about the CFD case
        casedata_file = [os.path.join(self.casepath,'CFD',file) for file in os.listdir(os.path.join(self.casepath,'CFD'))
                         if file == 'forces_breakdown.dat'][0]
        casedata = self.read_forces_breakdown_file(casedata_file)
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

        input_data_dir = os.path.join(self.casepath,'Postprocessing','BL_analysis','snapshots_bulkdata')
        sides = os.listdir(input_data_dir)
        output_data_dir = os.path.join(self.casepath,'Postprocessing','BL_analysis','snapshots_structured')
        if os.path.exists(output_data_dir):
            rmtree(output_data_dir)

        # Start structuring data
        print('Beginning to structure data...')
        for side in sides:
            print('--- Airfoil side: ' + side)

            snapshot_segments = os.listdir(os.path.join(input_data_dir,side))
            print('      Segment of analysis:')
            for x_c in snapshot_segments:  # loop over each segment
                print('      >> x/c ' + x_c)
                # Read input snapshot data
                snapshot_input_data_dir = os.path.join(input_data_dir, side, x_c)
                snapshot_input_data_files = [file for file in os.listdir(snapshot_input_data_dir) if file.endswith('.csv')]
                for snapshot_input_data_file in snapshot_input_data_files:  # loop over each snapshot defined in each segment
                    # Structure output snapshot data
                    snapshot_input_data = pd.read_csv(os.path.join(snapshot_input_data_dir,snapshot_input_data_file))
                    snapshot_output_data = snapshot_input_data[mapped_variables].copy()

                    if 'u' in self.snapshot_variables:
                        snapshot_output_data['u'] = snapshot_input_data['Momentum_0']*v_nd/snapshot_input_data['Density']
                    if 'v' in self.snapshot_variables:
                        snapshot_output_data['v'] = snapshot_input_data['Momentum_1']*v_nd/snapshot_input_data['Density']
                    if 'w' in self.snapshot_variables:
                        snapshot_output_data['w'] = snapshot_input_data['Momentum_2']*v_nd/snapshot_input_data['Density']
                    if 'P' in self.snapshot_variables:
                        snapshot_output_data['Pressure'] = snapshot_input_data['Pressure'] * Pr_nd
                    if 'T' in self.snapshot_variables:
                            snapshot_output_data['Temperature'] = snapshot_input_data['Temperature'] * T_nd
                    if 'rho' in self.snapshot_variables:
                        snapshot_output_data['Density'] = snapshot_input_data['Density'] * rho_nd

                    snapshot_output_data.columns = direct_variables + derived_variables

                    # Export output snapshot data
                    snapshot_output_data_dir = os.path.join(output_data_dir,side,x_c)
                    if not os.path.exists(snapshot_output_data_dir):
                        os.makedirs(snapshot_output_data_dir)

                    i_snapshot = int(re.search('_snapshot_(\d+).*', snapshot_input_data_file).group(1))
                    csvname = os.path.join(snapshot_output_data_dir, '%s_%s_DMD_snapshot_%d.csv' %(side,x_c,i_snapshot))
                    snapshot_output_data.to_csv(csvname, index=False, sep=',', decimal='.')

        print('Data structuring finished.')

    def dmd_analysis(self, export=False, compute_time_evolution=False):
        '''
        Method to perform the extraction of eigenvalues by DMD methods
        :param export: (bool) parameter to activate plots storage
        :param compute_time_evolution: (bool) parameter to activate the time evolution reconstruction from the extracted
        eigenvalues
        '''

        print('Beginning DMD analysis...')
        # Read data from reference case
        reference_data_files = [file for file in os.listdir(self.reference_data_case) if file.endswith('.dat')]
        reference_data = {'circle':None, 'physical': None}
        for file in reference_data_files:
            if 'circle' in file:
                reference_data['circle'] = pd.read_csv(os.path.join(self.reference_data_case,file),sep=' ').to_numpy()
            else:
                reference_data['physical'] = pd.read_csv(os.path.join(self.reference_data_case,file),sep=' ').to_numpy()
    
        Nvar = len(self.dmd_variables)
    
        # SET EXPORT FOLDER
        if export:
            export_dir = os.path.join(self.casepath,'Postprocessing','BL_analysis','DMD_analysis')
            if os.path.exists(export_dir):
                rmtree(export_dir)
            os.makedirs(export_dir)
    
        # Read snapshots
        snapshots_data_dir = os.path.join(self.casepath,'Postprocessing','Bl_analysis','snapshots_structured')
        sides = os.listdir(snapshots_data_dir)
        for side in sides:
            print('--- Airfoil side: ' + side)
            snapshot_data_segments_dir = os.listdir(os.path.join(snapshots_data_dir, side))
            dmd_eigs = [[]] *len(snapshot_data_segments_dir)

            print('      Segment of analysis:')
            for ix, x_c in enumerate(snapshot_data_segments_dir):  # loop over each segment
                print('      >> x/c ' + x_c)
                # Read input snapshot data
                snapshot_data_files_sorting_idx = [int(file.split('snapshot_')[1].split('.csv')[0]) for file in
                                        os.listdir(os.path.join(snapshots_data_dir, side, x_c)) if file.endswith('.csv')]
                snapshot_data_files = [file for file in os.listdir(os.path.join(snapshots_data_dir, side, x_c))
                                                   if file.endswith('.csv')]
                sorting_idx = np.argsort(snapshot_data_files_sorting_idx)
                snapshot_data_files = [snapshot_data_files[idx] for idx in sorting_idx]
    
                # Allocate dmd data container
                f = open(os.path.join(snapshots_data_dir,side,x_c,snapshot_data_files[0]),'r')
                reader = csv.reader(f, delimiter=',')
                Ndmd_x = len(snapshot_data_files)  # number of planes in chordwise direction
                Ndmd_z = len(list(reader)) - 1 # Number of points in normalwise direction
                snapshots = np.zeros([Ndmd_z*Nvar,Ndmd_x])
                x = np.zeros((Ndmd_x))
                for i, snapshot_data_file in enumerate(snapshot_data_files):  # loop over each snapshot defined in each segment
                    # Structure output snapshot data
                    snapshot_data = pd.read_csv(os.path.join(snapshots_data_dir, side, x_c, snapshot_data_file))
                    x[i] = snapshot_data['x'][0]
                    snapshots[:,i] = np.reshape(snapshot_data[self.dmd_variables].to_numpy().T,(Ndmd_z*Nvar,))
    
                # Determine snapshots x-spacing
                ds = x[1] - x[0]
    
                # FILTER SNAPSHOTS
                snapshots_filt = self.filter_nan(snapshots)
                Ndmd_x_filt = snapshots_filt.shape[1]
                #fig = plt.plot(svdvals(np.array([snapshot.flatten() for snapshot in snapshots_filt]).T), 'o')
    
                self.mu_list_manual = []
                self.lambda_list_manual = []
                self.W_list_manual = []
                self.mu_list_pydmd = []
                self.lambda_list_pydmd = []
                self.Psi_list_manual = []
                for arr in self.snapshots_arrangement.keys():
                    arrangement = np.arange(0,Ndmd_x_filt,int(arr),dtype=int)
                    snapshots_filt_arranged = snapshots_filt[:,arrangement]
    
                    # EIGENVALUE EXTRACTION PARAMETERS
                    rf = int(0.5*len(arrangement))
                    if rf > Ndmd_z:
                        rf = Ndmd_z
    
                    # EIGENVALUE EXTRACTION
                    # Manual procedure
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

                    self.mu_list_manual.append(mu_til_uc)
                    self.lambda_list_manual.append(mu_til)
                    self.W_list_manual.append(W_til)

                    # PyDMD procedure
                    dmd = DMD(svd_rank=rf, tlsq_rank=2, exact=True, opt=True)
                    dmd.fit(snapshots_filt)
                    dmd_eigs_uc = dmd.eigs
                    dmd_eigs = np.log(dmd.eigs)/ds
                    self.mu_list_pydmd.append(dmd_eigs_uc)
                    self.lambda_list_pydmd.append(dmd_eigs)

                    # TIME EVOLUTION COMPUTATION
                    if compute_time_evolution == True:
                        Phi = np.dot(np.dot(np.dot(X2,V[:,0:rf]),Sinv[0:rf,0:rf]),W_til)
                        b = np.dot(pinv(Phi),X[:,0])
                        Psi = np.zeros([rf,Ndmd_x], dtype='complex')
                        for i, _x in enumerate(x):
                            Psi[:,i] = np.multiply(np.power(mu_til,_x/ds),b)
                        self.Psi_list_manual.append(Psi)
    
                    '''
                    y = np.linspace(max(snapshot_data['z']),min(snapshot_data['z']),Ndmd_z)
                    fig, ax = plt.subplots(Nvar)
                    for ivar in range(Nvar):
                        ax[ivar].plot(X[ivar*Ndmd_z:Ndmd_z*(ivar+1)],y)
                    '''

                # PLOT
                self.plot_eigs(eigs=self.mu_list_manual,eigs_ref=reference_data['circle'],xc_segment=x_c,eig_type='uc',
                          computation='manual',export=True,export_dir=export_dir)
                self.plot_eigs(eigs=self.lambda_list_manual,eigs_ref=reference_data['physical'],xc_segment=x_c,eig_type='lambda',
                          computation='manual',dx=ds,export=True,export_dir=export_dir)

                self.plot_eigs(eigs=self.mu_list_pydmd,eigs_ref=reference_data['circle'],xc_segment=x_c,eig_type='uc',
                          computation='pydmd',export=True,export_dir=export_dir)
                self.plot_eigs(eigs=self.lambda_list_pydmd,eigs_ref=reference_data['physical'],xc_segment=x_c,eig_type='lambda',
                          computation='pydmd',dx=ds,export=True,export_dir=export_dir)
    
                print()

## -------------------------------------------------- INPUTS -------------------------------------------------------- ##
cases = {
    'NLF0416': {'casepath': r'C:\Users\juan.ramos\Altran\Proyectos\Transition\BL_transition\Cases\NLF0416_laminar',
                'reference_case_path': r'C:\Users\juan.ramos\Altran\Proyectos\Transition\BL_transition\Cases\NLF0416_laminar\Postprocessing\Reference_data'
                }
}

dmd_parameters = {
    'dmd_variables': ['u','w'],
    'snapshot_variables': ['x', 'z', 'u', 'z'],
    'snapshot_arrangement': {'1':'r','2':'b','3':'c'},
    'analysis_segments':
                        [
                        0.1,
                        0.2,
                        0.3,
                        0.4,
                        0.5,
                        0.6,
                        0.7,
                        0.8,
                        0.9,
                        ],
}

grid_parameters = {
'wall_distance': 'delta0',
'deltaT': 0.01,
'delta0':8e-06,
'GR':1.1,
'NL': 40,
'DX': None, 
'N1': 20}

for (ID, case) in cases.items():
    DMD_analyzer = DMD_scanner(case,grid_parameters,dmd_parameters)
    DMD_analyzer.generate_snapshot_grid()
    DMD_analyzer.generate_snapshot_data(dymform='D')
    DMD_analyzer.dmd_analysis(export=True)