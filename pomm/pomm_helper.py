#------------------------------------------------------------------------------- SCRIPT INFO
'''
File name:              helper.py
Author:                 Oliver Carmignani
Date of creation:       02/28/2023
Date last modified:     02/28/2023
Python Version:         3.10
'''


#------------------------------------------------------------------------------- MODULES
from datetime import  datetime, timezone, timedelta, date
import pandas as pd
import warnings
import numpy as np
from numba import jit, njit
import netCDF4 as nc
import os
from skimage import io, color
from sklearn.metrics import r2_score


#------------------------------------------------------------------------------- SETTINGS
## IGNORE WARNINGS IN JUPYTER NOTEBOOK
warnings.filterwarnings('ignore')


#------------------------------------------------------------------------------- FUNCTIONS
## TIME FEATURES
def create_timefeatures(df):
    """
    Create time series features based on time series index
    """
    df = df.copy()
    df['minute'] = df.index.minute
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.day_of_week
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    
    return df


def calc_Q(a, b, H, H0):
    """
    """
    return a * (H - H0) ** b


def catchment_size(catchment, shape_0, shape_1):
    """
    """
    # Empty list to capture indices
    x_minmax = []
    y_minmax = []

    # Iterate through rows, get rowsums > 0
    for i in range(shape_0):
        if np.sum(catchment[i,:]) > 0:
            x_minmax.append(i)

    # Iterate through columns, get get colsums > 0
    for j in range(shape_1):
        if np.sum(catchment[:,j]) > 0:
            y_minmax.append(j)

    # Dimensions are given bi first and last captured indices
    x_min, x_max = x_minmax[0]-1, x_minmax[-1]+1
    y_min, y_max = y_minmax[0]-1, y_minmax[-1]+1
    if y_min < 0: y_min = 0
    if x_min < 0: x_min = 0

    return x_min, x_max, y_min, y_max


# Read netcdf file
def read_nc(path, info=True):
    """
    """
    ds = nc.Dataset(path)
    if info==True:
        ## PRINT GENERAL OVERVIEW
        print(f'\nDATASET{"":-^100}\n{ds}')
        print(f'\nDICTIONARY{"":-^100}\n{ds.__dict__}')
        print(f'\nDICT KEYS{"":-^100}\n{ds.__dict__.keys()}')
        print(f'\nDIMENSIONS{"":-^100}')
        [print(dim) for dim in ds.dimensions.values()]
        print(f'\nVARIABLES{"":-^100}\n')
        [print(f'{key}:\n {ds.variables[key]}\n') for key in ds.variables.keys()]
        print(f'\nVARIABLE KEYS{"":-^100}\n{ds.variables.keys()}')
        print(f'\nLAT/LON DISTRIBUTION{"":-^100}')
        print(f'lat:\n{ds.variables["latitude"][:]}\n\nlon:\n{ds.variables["longitude"][:]}')
    lat_size = len(ds.variables["latitude"][:])
    lon_size = len(ds.variables["longitude"][:])
    return ds, lat_size, lon_size


# Load ERA5 files
def load_ERA5(lat_index, lon_index, months, years, plot_nc, plant_name, subdir=None):
    """
    """
    ## INITIAL VARIABLES
    # START = datetime.strptime('1900-01-01 00:00:00.0', '%Y-%m-%d %H:%M:%S.%f')
    df_era5 = pd.DataFrame()

    ## LOOP OVER YEARS AND MONTHS
    for i, year in enumerate(years):
        for j, month in enumerate(months):

            ## SHOW INFO ONLY FOR FIRST FILE
            if i==0 and j==0: info = True
            else: info = False
            
            ## READ CURRENT NC-FILE
            try:
                if not subdir is None:
                    ds, lat_size, lon_size = read_nc('./data/ERA5/' + subdir + plant_name + '_ERA5_' + year + '_' + month + '.nc', info=info)
                else:
                    ds, lat_size, lon_size = read_nc('./data/ERA5/' + plant_name + '_ERA5_' + year + '_' + month + '.nc', info=info)
            except Exception as e:
                print(e)
                print('Continue executed.')

            ## PRINT LON/LAT INFO
            if i==0 and j==0:
                ## SHOW MAP
                plot_nc(ds, lat_size, lon_size)
                lat = round(float(ds.variables["latitude"][lat_index]),2)
                lon = round(float(ds.variables["longitude"][lon_index]),2)
                print(f'Chosen lat/lon: {lat} {lon}')

            ## CONVERT TIMESTAMPS FROM 'hours since...' TO DATETIME FORMAT
            try:
                times = np.array(ds.variables['time']) # Old ERA5 implementation --> Time variable was called 'time'
                print('Old ERA5 implementation used')
                START = datetime.strptime('1900-01-01 00:00:00.0', '%Y-%m-%d %H:%M:%S.%f')
                leave_out_variables = 3
                Date = [START + timedelta(hours=int(times[i])) for i in range(times.shape[0])]
            except Exception as e:
                print('\nFrigg: New ERA5 implementation detected, causing the following Error:')
                print(e)
                print('Frigg: New variable name for time used instead.')
                print('Frigg: New hours since starting points used instead.')
                times = np.array(ds.variables['valid_time']) # New ERA5 implementation --> Time variable is now called 'valid_time'
                START = datetime.strptime('1970-01-01 00:00:00.0', '%Y-%m-%d %H:%M:%S.%f')
                leave_out_variables = 5
                Date = [START + timedelta(seconds=int(times[i])) for i in range(times.shape[0])]

            ## PULL OUT VARIABLES, CONVERT TO NUMPY TS
            variables = list(ds.variables.keys())[leave_out_variables:]
            values = []
            for variable in variables:
                values.append(np.array(ds.variables[variable])[:,lat_index,lon_index])

            ## CREATE PANDAS DATAFRAME
            df_tmp = pd.DataFrame({
                'date': Date,
            })
            for k in range(len(variables)):
                df_tmp[variables[k]] = values[k]

            ## SET DATE AS INDEX, ADD TIME FEATURES FOR LATER BOXPLOTS
            df_tmp = df_tmp.set_index('date')
            df_tmp = create_timefeatures(df_tmp)

            ## CREATE OR APPEND DATAFRAME
            if i==0 and j==0:
                df_era5 = df_tmp.copy()
            else:
                df_era5 = df_era5._append(df_tmp)

    ## SORT INDEX
    df_era5 = df_era5.sort_index()

    return df_era5, lat, lon


def calc_nse(mod, val):
    """
    """
    return 1 - ( sum( (mod-val)**2 ) / sum( (val-np.mean(val))**2 ) )


def calc_mse(mod, val, n):
    """
    """
    return np.sum(np.sqrt((mod-val)**2)) / n


@njit(fastmath=True)
def calc_distance(catchment, flowdir, intake_x, intake_y, res):

    ## RESERVE MATRICES
    amount = 1000 # Maximum amount of steps
    movementX = np.zeros((amount, catchment.shape[0], catchment.shape[1]), dtype=np.int32) # 3D-Matrix with time-depended movement in X-direction
    movementY = np.zeros((amount, catchment.shape[0], catchment.shape[1]), dtype=np.int32) # 3D-Matrix with time-depended movement in Y-direction
    steps_to_intake = np.zeros((catchment.shape[0], catchment.shape[1]), dtype=np.int32) # 2D-Matrix with how many steps each pixxel took to reach intake
    distance_to_intake = np.zeros((catchment.shape[0], catchment.shape[1]), dtype=np.int32) # 2D-Matrix with distance to intake in m for each pixxel
    accumulation = np.ones((catchment.shape[0], catchment.shape[1]), dtype=np.int32) # Calculate how many pixxels are "flowing" through this position

    ## PRE SETTINGS / INFOS
    tot = np.sum(catchment) # Pixxels to calculate
    print('Pixxels to calculate:', tot)
    lv = 1 # Pixxels calculated
    z = -1 # Target reached after z steps

    ## LOOP UNTIL ALL PIXXELS REACHED INTAKE
    while True:

        ## CHECK IF RESERVED MATRICE-SIZE REACHED
        z += 1
        if z == amount:
            print(lv, tot, z)
            return movementX, movementY, steps_to_intake, distance_to_intake, z, accumulation

        ## CALC FOR EACH PIXXEL NEXT STEP
        for x in range(catchment.shape[0]):
            for y in range(catchment.shape[1]):
                
                ## CHECK IF CURRENT PIXXEL IS WITHIN CATCHMENT
                if not catchment[x,y] > 0:
                    continue

                ## HANDLE FIRST STEP DIFFERENT
                if z > 0:
                    i = int(movementX[z-1,x,y])
                    j = int(movementY[z-1,x,y])
                    ## CHECK IF CURRENT PIXXEL HAS REACHED INTAKE
                    if i == intake_x and j == intake_y:
                        steps_to_intake[x,y] = z

                        lv += 1
                        ## CHECK IF ITS LAST PIXXEL WHICH REACHED INTAKE, RETURN IF SO
                        if lv == tot:
                            print('All pixxels done at step', z)
                            print('Hydraulic Length:', np.max(distance_to_intake), 'm')
                            # movementX = movementX[:z,:,:]
                            # movementY = movementY[:z,:,:]
                            return movementX, movementY, steps_to_intake, distance_to_intake, z, accumulation
                        continue
                else:
                    i = x
                    j = y

                ## CHECK IF MOVEMENT WOULD FLOW OUT OF PICTURE, THEN PASS
                if ((i+1) >= catchment.shape[0]) or ((j+1) >= catchment.shape[1]) or (i < 0) or (j < 0):
                    continue

                ## CHECK FLOWDIRECTION
                # North East    1
                if flowdir[i,j] == 1:
                    movementX[z,x,y] = i-1
                    movementY[z,x,y] = j+1
                    distance_to_intake[x,y] += np.sqrt(res**2 + res**2)
                    accumulation[i-1,j+1] += 1
                # East          8
                elif flowdir[i,j] == 8:
                    movementX[z,x,y] = i
                    movementY[z,x,y] = j+1
                    distance_to_intake[x,y] += res
                    accumulation[i,j+1] += 1
                # South East    7
                elif flowdir[i,j] == 7:
                    movementX[z,x,y] = i+1
                    movementY[z,x,y] = j+1
                    distance_to_intake[x,y] += np.sqrt(res**2 + res**2)
                    accumulation[i+1,j+1] += 1
                # South         6
                elif flowdir[i,j] == 6:
                    movementX[z,x,y] = i+1
                    movementY[z,x,y] = j
                    distance_to_intake[x,y] += res
                    accumulation[i+1,j] += 1
                # South West    5
                elif flowdir[i,j] == 5:
                    movementX[z,x,y] = i+1
                    movementY[z,x,y] = j-1
                    distance_to_intake[x,y] += np.sqrt(res**2 + res**2)
                    accumulation[i+1,j-1] += 1
                # West          4
                elif flowdir[i,j] == 4:
                    movementX[z,x,y] = i
                    movementY[z,x,y] = j-1
                    distance_to_intake[x,y] += res
                    accumulation[i,j-1] += 1
                # North West    3
                elif flowdir[i,j] == 3:
                    movementX[z,x,y] = i-1
                    movementY[z,x,y] = j-1
                    distance_to_intake[x,y] += np.sqrt(res**2 + res**2)
                    accumulation[i-1,j-1] += 1
                # North         2
                elif flowdir[i,j] == 2:
                    movementX[z,x,y] = i-1
                    movementY[z,x,y] = j
                    distance_to_intake[x,y] += res
                    accumulation[i-1,j] += 1
                # Nothing
                else:
                    continue


def export_blender(elevation, tensor):
    """
    """
    ## EXPORT ELEVATION CUTTED
    if not os.path.isdir('./Blender'): os.mkdir('./Blender')
    if not os.path.isdir('./Blender/water_added'): os.mkdir('./Blender/water_added')
    io.imsave('./Blender/elevation.tif', elevation)


    ## EXPORT
    for i in range(tensor.shape[0]):
        ## EXPORT STATUS
        if i % 100 == 0:
            print(f'{i} / {tensor.shape[0]}')
        tmp = np.copy(elevation)
        io.imsave('./Blender/water_added/storage_' + str(i) + '.tif', tmp + tensor[i,:,:])
    

#------------------------------------------------------------------------------- MAIN RUN
if __name__ == '__main__':
    print('pomm_helper main run')