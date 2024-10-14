#------------------------------------------------------------------------------- SCRIPT INFO
'''
File name:              pomm.py
Author:                 Oliver Carmignani
Date of creation:       03/03/2023
Date last modified:     03/03/2023
Python Version:         3.10
'''


#------------------------------------------------------------------------------- MODULES
# Standart python libraries
import os
import numpy as np
import json
# Handeling of geotiff files
from skimage import io
# Computation speed
from numba import njit
# Pomm library
if __name__ == '__main__':
    from pomm_helper import *
    from pomm_vis import *
else:
    from pomm.pomm_helper import *
    from pomm.pomm_vis import *


#------------------------------------------------------------------------------- MODEL
class Pomm():

    def __init__(self, station_name='Example'):
        '''
        '''
        ## PATHS
        self.path = '\\'.join(os.path.dirname(os.path.abspath(__file__)).split('\\')[:-1])
        self.input =  self.path + '\\input\\'
        self.output = self.path + '\\output\\'


        ## CONSTANTS
        self.station_name = station_name
        self.intake_x, self.intake_y = 0, 0
        self.spatial_res = 30
        self.temporal_res = 'h'
        self.soil_bands = [3,4,5]

#_______________________________________________________________________________ MODEL PRE-PROCESS

    def load_attributes(self):
        """
        """
        ## LOAD DATA FROM attributes.json
        try:
            with open( './attributes.json') as f:
                self.params = json.load(f)
            self.intake_x, self.intake_y = self.params[self.station_name]['Intake'][0], self.params[self.station_name]['Intake'][1]
            self.spatial_res = self.params[self.station_name]['Resolution'] # in Meters (m)
            self.coordinates = self.params[self.station_name]['Coordinates']
        except Exception as e:
            print('Error')
            print(e)

    def generate_example(self, size):
        """
        """
        # Generate Surface
        xv = np.linspace(-5, 5, size, dtype=np.float32)
        yv = np.linspace(0, 5, size, dtype=np.float32)
        x,y = np.meshgrid(xv, yv)
        self.elevation = (9-x**2 - y**2) * -10 + 100

        # Generate Flowdirection
        #N    NE    E    SE    S    SW    W    NW
        # dirmap = (64,  128,  1,   2,    4,   8,    16,  32)
        dirmap = (3,  2,  1,   8,    7,   6,    5,  4)
        self.flowdir = np.zeros((self.elevation.shape[0],self.elevation.shape[1]))
        for i in range(self.elevation.shape[0]):
            for j in range(self.elevation.shape[1]):
                # Borders
                if j == 0:
                    # self.flowdir[i,j] = 32
                    self.flowdir[i,j] = 4
                elif i == 0:
                    # self.flowdir[i,j] = 128
                    self.flowdir[i,j] = 2
                elif j == self.elevation.shape[1]-1:
                    # self.flowdir[i,j] = 2
                    self.flowdir[i,j] = 8
                elif i == self.elevation.shape[0]-1:
                    # self.flowdir[i,j] = 8
                    self.flowdir[i,j] = 6
                # Centers
                else:
                    # Surrounding pixxels with its distances
                    surroundings = np.array([self.elevation[i-1,j-1], self.elevation[i-1,j], self.elevation[i-1,j+1],
                                    self.elevation[i,j+1], self.elevation[i+1,j+1], self.elevation[i+1,j],
                                    self.elevation[i+1,j-1], self.elevation[i,j-1]])
                    distance = self.elevation[i,j] - surroundings
                    # Take the highest values
                    largest = np.max(distance)
                    choice = np.asarray(np.where(distance == largest))
                    # Just take the first entry of choice
                    self.flowdir[i,j] = dirmap[choice[0][0]]

        # Generate accumulation
        self.accumulation = np.zeros([size, size])
        self.accumulation_v = np.zeros([size, size])
        self.catchment = np.ones([size, size])
        self.soil = np.zeros([12,size, size])
        self.soil_v = np.zeros([size, size])
        self.num_pixxel = np.sum(self.catchment)

    def load_input(self, num_c=1, soil_years=['2022']):
        '''
        '''
        ## ELEVATION
        print(f'{"":-^100}')
        print(f'Load elevation:\t\t', end=' ')
        try:
            self.elevation = io.imread(self.input + 'elevation\\' + self.station_name + '_filled.tif')
            print('success')
        except Exception as e:
            print('Error')
            print(e)

        ## FLOWDIRABS
        print(f'Load flowdirections:\t', end=' ')
        try:
            self.flowdir = io.imread(self.input + 'elevation\\' + self.station_name + '_flowdirabs.tif')
            print('success')
        except Exception as e:
            print('Error')
            print(e)

        ## ACCUMULATION
        print(f'Load accumulation:\t', end=' ')
        try:
            self.accumulation = io.imread(self.input + 'elevation\\' + self.station_name + '_accumulation.tif')
            # Change values of accumulation for better view (river system)
            self.accumulation_v = self.accumulation.copy()
            self.accumulation_v = np.nan_to_num(self.accumulation_v)
            self.accumulation_v[self.accumulation_v<0] *= -1
            self.accumulation_v = np.sqrt(np.sqrt(self.accumulation_v))
            print('success')
        except Exception as e:
            print('Error')
            print(e)

        ## CATCHMENT
        print(f'Load catchment:\t\t', end=' ')
        try:
            self.catchments = [None] * num_c
            self.num_pixxels = [None] * num_c
            for i in range(len(self.catchments)):
                self.catchments[i] = io.imread(self.input + 'catchment\\' + self.station_name + '_catchment' + str(i+1) + '.tif')
                ## MAKE SURE THAT CATCHMENT-PIXXELS == 1 AND NO-CATCHMENT-PIXXELS == 0
                self.catchments[i][np.where(self.catchments[i]==255)] = 0
                self.catchments[i][np.where(self.catchments[i]>0)] = 1
                self.num_pixxels[i] = self.catchments[i].sum()
            self.catchment = self.catchments[0]
            self.num_pixxel = self.num_pixxels[0]
            print('success')
        except Exception as e:
            print('Error')
            print(e)

        ## SOIL
        print(f'Load soil:\t\t', end=' ')
        try:
            self.soils = [None] * len(soil_years)
            self.soils_v = [None] * len(soil_years)
            for i in range(len(self.soils)):
                ## ORIGINAL SOIL-FILE WITH ALL BANDS
                self.soils[i] = io.imread(self.input + 'soil\\' + self.station_name + '_soil_' + soil_years[i] + '.tif')
                # Traspose axis for better interaction with different bands
                self.soils[i] = np.transpose(self.soils[i], (2,0,1))
                ## GENERATE SOIL PICTURE WITH MEDIAN OF CHOSEN BANDS ACCORDING TO GOOGLE-EARTH-ENGINE
                self.soils_v[i] = self.soils[i].copy()
                self.soils_v[i][np.where(self.soils_v[i]<0)] = np.nan
                band_index = [self.soil_bands[i]-1 for i in range(len(self.soil_bands))]
                self.soils_v[i] = np.median(self.soils_v[i][band_index,:,:], axis=0)
            self.soil = self.soils[0]
            self.soil_v = self.soils_v[0]
            print('success')
        except Exception as e:
            print('Error')
            print(e)

    def catchment_cut(self):
            """
            """
            print(f'{"":-^100}')
            print('Init Shape:\t\t', self.catchment.shape)

            ## GET INDICES OF CATCHMENT AREA WITHIN TIF FILE
            self.x_min, self.x_max, self.y_min, self.y_max = catchment_size(self.catchment, self.catchment.shape[0], self.catchment.shape[1])
            print('Indices:\t\t', self.x_min, self.x_max, self.y_min, self.y_max)

            ## CUT FILES
            self.catchment_cutted = self.catchment.copy()
            self.elevation_cutted = self.elevation.copy()
            self.flowdir_cutted = self.flowdir.copy()
            self.soil_cutted = self.soil.copy()
            self.soil_v_cutted = self.soil_v.copy()
            self.accumulation_cutted = self.accumulation.copy()
            self.accumulation_v_cutted = self.accumulation_v.copy()
            self.catchment_cutted = self.catchment_cutted[self.x_min:self.x_max,self.y_min:self.y_max]
            self.elevation_cutted = self.elevation_cutted[self.x_min:self.x_max,self.y_min:self.y_max]
            self.flowdir_cutted = self.flowdir_cutted[self.x_min:self.x_max,self.y_min:self.y_max]
            self.soil_cutted = self.soil_cutted[:,self.x_min:self.x_max,self.y_min:self.y_max]
            self.soil_v_cutted = self.soil_v_cutted[self.x_min:self.x_max,self.y_min:self.y_max]
            self.accumulation_cutted = self.accumulation_cutted[self.x_min:self.x_max,self.y_min:self.y_max]
            self.accumulation_v_cutted = self.accumulation_v_cutted[self.x_min:self.x_max,self.y_min:self.y_max]

            # ## SET NON CATCHMENT AREA PIXXELS TO NAN OR ZERO (depends on usecase)
            # self.elevation_cutted = np.where(self.catchment_cutted==0, np.nan, self.elevation_cutted)
            # self.flowdir_cutted[np.where(self.catchment_cutted==0)] = 0
            # self.soil_cutted = np.where(self.catchment_cutted==0, np.nan, self.soil_cutted)
            # self.soil_v_cutted = np.where(self.catchment_cutted==0, np.nan, self.soil_v_cutted)
            # self.accumulation_cutted[np.where(self.catchment_cutted==0)] = 0
            # self.accumulation_v_cutted[np.where(self.catchment_cutted==0)] = 0

            ## OUTPUT SIZE
            print('New Shape:\t\t', self.catchment_cutted.shape)

    def hydraulic_length(self, amount=1000):
        """
        """
        ## CALC ACCUMULATION
        self.movementX, self.movementY, self.steps_to_intake, self.distance_to_intake, z, self.accumulation_calculated = calc_distance(self.catchment_cutted, self.flowdir_cutted, self.intake_x, self.intake_y, self.spatial_res, amount)
        ## CUT RESERVED MATRIX TO MAX STEP LENGTH
        self.movementX, self.movementY = self.movementX[:z], self.movementY[:z]

        self.accumulation_calculated[self.catchment_cutted==0] = 0

        ## GET INDICES AND PATH OF PIXXEL WITH MOST STEPS
        self.stepsMaxX = np.where(self.steps_to_intake==np.max(self.steps_to_intake))[0]
        self.stepsMaxY = np.where(self.steps_to_intake==np.max(self.steps_to_intake))[1]
        self.stepsMaxPath = np.zeros_like(self.steps_to_intake)
        try:
            self.stepsMaxPath[self.movementX[:,self.stepsMaxX,self.stepsMaxY],self.movementY[:,self.stepsMaxX,self.stepsMaxY]] = 1
        except Exception as e:
            print(e)


        ## GET INDICES AND PATH OF PIXXEL WITH LONGEST PATHH (Hydraulic length)
        self.distanceMaxX = np.where(self.distance_to_intake==np.max(self.distance_to_intake))[0]
        self.distanceMaxY = np.where(self.distance_to_intake==np.max(self.distance_to_intake))[1]
        self.distanceMaxPath = np.zeros_like(self.distance_to_intake)
        self.distanceMaxPath[self.movementX[:,self.distanceMaxX,self.distanceMaxY],self.movementY[:,self.distanceMaxX,self.distanceMaxY]] = 1

    def river_hirarchy(self):
        """
        """
        self.rs = np.copy(self.accumulation_cutted)
        self.rs_v = np.copy(self.accumulation_v_cutted)

        # Main River --> Everything above 0.5*max value in self.accumulation_v_cutted
        # Main River --> Set Value to 5
        self.rs = self.rs*0
        th_main = 0.5 * np.max(self.accumulation_cutted)
        self.rs[np.where(self.accumulation_cutted>th_main)] = 5
        self.rs_v = self.rs_v*0
        th_main_v = 0.5 * np.max(self.accumulation_v_cutted)
        self.rs_v[np.where(self.accumulation_v_cutted>th_main_v)] = 5

        # Secondary River --> Everything above 0.2*max value in self.accumulation_v_cutted AND below 0.5*max value in self.accumulation_v_cutted
        # Secondary River --> Set Value to 2
        th_sec = 0.2 * np.max(self.accumulation_cutted)
        self.rs[np.where(np.logical_and(self.accumulation_cutted>th_sec, self.accumulation_cutted<=th_main))] = 2
        th_sec_v = 0.2 * np.max(self.accumulation_v_cutted)
        self.rs_v[np.where(np.logical_and(self.accumulation_v_cutted>th_sec_v, self.accumulation_v_cutted<=th_main_v))] = 2

        # Within Catchment --> Values == 0
        # Out of Catchment --> Values == -1
        self.rs[np.where(self.catchment_cutted==0)] = -1
        self.rs_v[np.where(self.catchment_cutted==0)] = -1

    def Generate_CN(self, CN_borders):
        """
        """
        # self.CN_choice = np.array([50,55,60,65,70,75]) # NSE 0.41
        # self.CN_choice = np.array([50,55,60,65,70,95]) # NSE 0.48
        self.CN_choice = np.array([50,60,70,80,90,100])
        # self.CN_choice = np.array([50,58,60,65,70,95]) # NSE 0.484
        # self.CN_borders = np.array([200, 400, 600, 800])
        self.CN_borders = CN_borders
        self.CN_init = np.copy(self.soil_v_cutted)
        self.CN_init[np.where(self.soil_v_cutted <= self.CN_borders[0])] = self.CN_choice[1]
        self.CN_init[np.where(np.logical_and(self.soil_v_cutted>self.CN_borders[0], self.soil_v_cutted<=self.CN_borders[1]))] = self.CN_choice[2]
        self.CN_init[np.where(np.logical_and(self.soil_v_cutted>self.CN_borders[1], self.soil_v_cutted<=self.CN_borders[2]))] = self.CN_choice[3]
        self.CN_init[np.where(np.logical_and(self.soil_v_cutted>self.CN_borders[2], self.soil_v_cutted<=self.CN_borders[3]))] = self.CN_choice[4]
        self.CN_init[np.where(self.soil_v_cutted > self.CN_borders[3])] = self.CN_choice[5]
        self.CN_init[np.where(self.catchment_cutted==0)] = self.CN_choice[0]


        print(f'CN\tOCC\tPERC')
        for cn, occ in zip(np.unique(self.CN_init, return_counts=True)[0], np.unique(self.CN_init, return_counts=True)[1]):
            if not np.isnan(cn):
                print(f'{cn}\t{occ}\t{round(occ/self.num_pixxel*100, 1)} %')

    def Generate_Sabs(self, Sabs_borders):
        """
        """
        ## Transform to Sabs
        self.Sabs_choice = np.array([1,50,40,30,20,10])
        # self.Sabs_borders = np.array([200, 400, 600, 800])
        self.Sabs_borders = Sabs_borders
        self.Sabs_init = np.copy(self.soil_v_cutted)
        self.Sabs_init[np.where(self.soil_v_cutted <= self.Sabs_borders[0])] = self.Sabs_choice[1]
        self.Sabs_init[np.where(np.logical_and(self.soil_v_cutted>self.Sabs_borders[0], self.soil_v_cutted<=self.Sabs_borders[1]))] = self.Sabs_choice[2]
        self.Sabs_init[np.where(np.logical_and(self.soil_v_cutted>self.Sabs_borders[1], self.soil_v_cutted<=self.Sabs_borders[2]))] = self.Sabs_choice[3]
        self.Sabs_init[np.where(np.logical_and(self.soil_v_cutted>self.Sabs_borders[2], self.soil_v_cutted<=self.Sabs_borders[3]))] = self.Sabs_choice[4]
        self.Sabs_init[np.where(self.soil_v_cutted > self.Sabs_borders[3])] = self.Sabs_choice[5]
        self.Sabs_init[np.where(self.catchment_cutted==0)] = self.Sabs_choice[0]


        print(f'Sabs\tOCC\tPERC')
        for sabs, occ in zip(np.unique(self.Sabs_init, return_counts=True)[0], np.unique(self.Sabs_init, return_counts=True)[1]):
            if not np.isnan(sabs):
                print(f'{sabs}\t{occ}\t{round(occ/self.num_pixxel*100, 1)} %')

    def Generate_MC(self, mcs_overland=[2000,1000], mcs_subsurface=[3600,2000]):
        """
        """
        ## Transform to mc_overland
        self.mc_overland = np.ones((self.x_max-self.x_min, self.y_max-self.y_min))
        a = (np.log(mcs_overland[1]) - np.log(mcs_overland[0])) / (50)
        b = np.log(mcs_overland[0]) - 50 * a
        self.mc_overland = np.exp(a * self.CN_init + b)
        ## Show MC-CN-Distribution
        ts = np.exp(a * np.arange(0,100) + b)
        ts[0:50] = mcs_overland[1]
        # plot_mc_cn(ts=ts, CN_borders=np.nan_to_num(self.CN_choice).astype('int'))

        ## Transform to mc_subsurface
        self.mc_subsurface = np.ones((self.x_max-self.x_min, self.y_max-self.y_min))
        a_soil = (np.log(mcs_subsurface[1]) - np.log(mcs_subsurface[0])) / (50)
        b_soil = np.log(mcs_subsurface[0]) - 50 * a_soil
        self.mc_subsurface = np.exp(a_soil * self.CN_init + b_soil)
        ## Show MC-CN-Distribution
        ts = np.exp(a_soil * np.arange(0,100) + b_soil)
        ts[0:50] = mcs_subsurface[1]
        # plot_mc_cn(ts=ts, CN_borders=np.nan_to_num(self.CN_choice).astype('int'))

#_______________________________________________________________________________ MODEL PROCESS

    @staticmethod
    @njit(fastmath=True)
    def pomm_runoff(
        ## Rainfall
        precip,
        ## GeoSpacial
        catchment, elevation, spatial_res, flowdir, x_min, x_max, y_min, y_max, T, intake_x, intake_y,
        ## Pomm Specific
        CN_init, Sabs_init, mc_overland, mc_subsurface, evap_factors, evap_month,
        ## Checkpoint
        water, soil, Q_water, Q_soil, V_water, V_soil
        ):
        """
        """


        ## INITIALIZE
        # Size
        size_x = catchment.shape[0]
        size_y = catchment.shape[1]

        # Water
        water_storage = np.zeros((T+1,size_x,size_y), dtype=np.float32)
        water_storage[0,:,:] = water
        # Overlandwater in the system
        water_tot = np.zeros(T, dtype=np.float32)
        water_latest = 0

        # Soil
        soil_storage = np.zeros((T+1,size_x,size_y), dtype=np.float32)
        soil_storage[0,:,:] = soil
        # Soilwater in the system
        soil_tot = np.zeros(T, dtype=np.float32)
        soil_latest = 0

        # Absorbtion
        Sabs = Sabs_init

        # Filtration into soil
        F = np.zeros((size_x,size_y), dtype=np.float32)

        # Curve Number
        CN = CN_init
        # Soil conservation, [mm --> m]
        S = 1 * ((2540 / CN - 25.4) / 1000 )
        # Evolution parameter of CN
        n = 1
        # small adjustment for the evolution to avoid division by 0
        epsilon = 0.001
        # constant for CN evolution
        a = np.ones((size_x,size_y), dtype=np.float32)
        a = 1 / ( (CN / 100)**n * (100 + epsilon - CN)**(n+1) / epsilon**n )

        # Initial abstraction
        # Lambda, used for initial abstraction, [0, 0.4]
        lam = np.ones((size_x,size_y), dtype=np.float32)
        lam = lam * 0.4
        I = np.zeros((size_x,size_y), dtype=np.float32)
        I = lam * S

        # Gravitational force, m/s**2
        g = 9.81

        # time
        delta_t = 60

        # Slope
        s_water = np.zeros((size_x,size_y), dtype=np.float32)
        s_soil = np.zeros((size_x,size_y), dtype=np.float32)

        # Mannings coefficient
        mc_water = mc_overland
        mc_soil = mc_subsurface
        mc_factor = np.ones(T, dtype=np.float32)

        # Max water shift
        # max_shift = 500
        max_shift = 10000000000000000

        # Evaporation
        evapo = 0.5

##########################################################################################

        for t in range(T):

            # PRINT STATUS
            if t % 100 == 0 or t == 0:
                print(t)

            ## UPDATES AFTER EACH TIME-STEP
            # Water in the system update
            water_tot[t] = water_latest
            soil_tot[t] = soil_latest
            # Temporary Water Matrix
            water_tmp = np.zeros((size_x,size_y), dtype=np.float32)
            soil_tmp = np.zeros((size_x,size_y), dtype=np.float32)
            # Rain Matrix
            evapo = evap_factors[evap_month[t%12]]
            rain = np.ones((size_x,size_y), dtype=np.float32) * precip[t]
            rain *= catchment # Only let it rain over catchment pixxels
            rain *= evapo
            water += rain

##########################################################################################

            ## FILTRATION and INFILTRATION
            for i in range(1, size_x-1):
                for j in range(1, size_y-1):

                    ## INITIAL CONDITIONS
                    # If current pixxel not in catchment, pass
                    if catchment[i,j] == 0:
                        water[i,j] = 0
                        continue

                    ## INITIAL ABSTRACTION
                    if rain[i,j] == 0:
                        I[i,j] = 0

                    ## FILTRATION
                    # Initial abstraction, Filtration
                    if rain[i,j] > I[i,j]:
                        # Filtration
                        comp1 = a[i,j] * (rain[i,j] - I[i,j]) * (CN_init[i,j]/100)**n * (100 - CN[i,j])**(n+1) / (CN[i,j] - CN_init[i,j] + epsilon)**n
                        comp2 = Sabs[i,j] - soil[i,j]
                        if comp1 <= comp2:
                            F[i,j] = comp1
                        else:
                            F[i,j] = comp2
                        if F[i,j] < 0:
                            F[i,j] = 0
                        ## SOIL
                        # If Sabs is reached --> filtration stops..
                        comp1 = 0
                        comp2 = soil[i,j] + F[i,j]
                        if comp1 >= comp2:
                            soil[i,j] = comp1
                        else:
                            soil[i,j] = comp2

                        # Check if soil is full
                        if soil[i,j] >= Sabs[i,j]:
                            # Water on top
                            comp1 = 0
                            comp2 = water[i,j] - I[i,j] - F[i,j] + (soil[i,j] - Sabs[i,j]) # precip already added !
                            if comp1 >= comp2:
                                water[i,j] = comp1
                            else:
                                water[i,j] = comp2

                            # If soil is full it needs to be deducted
                            # soil[i,j] = soil[i,j] - (soil[i,j] - Sabs)
                            soil[i,j] = Sabs[i,j]
                        else:
                            # Water on top
                            comp1 = 0
                            comp2 = water[i,j] - I[i,j] - F[i,j] # precip already added !
                            if comp1 >= comp2:
                                water[i,j] = comp1
                            else:
                                water[i,j] = comp2
                            # Soil
                            soil[i,j] = soil[i,j]
                    else:
                        # Filtration
                        F[i,j] = 0
                        # Water on top
                        comp1 = 0
                        comp2 = water[i,j] - rain[i,j]
                        if comp1 >= comp2:
                            water[i,j] = comp1
                        else:
                            water[i,j] = comp2
                        # Check if soil is full
                        if soil[i,j] >= Sabs[i,j]:
                            # If soil is full it needs to be deducted
                            # soil[i,j] = soil[i,j] - (soil[i,j] - Sabs)
                            soil[i,j] = Sabs[i,j]

##########################################################################################

            ## ENERGY HEAD
            E_water = elevation + water + 1.0 / (2 * g) * V_water**2
            E_soil = elevation + soil + 1.0 / (2 * g) * V_soil**2

            # Empty outflow pixxel
            water[intake_x,intake_y] = 0
            E_water[intake_x,intake_y] = 0
            E_soil[intake_x,intake_y] = 0

##########################################################################################

            ## RUNOFF
            for i in range(1, size_x-1):
                for j in range(1, size_y-1):

                    ## INITIAL CONDITIONS
                    # If current pixxel not in catchment, pass
                    if catchment[i,j] == 0:
                        water[i,j] = 0
                        continue

                    #################################################################################
                    ## OVERLAND

                    ## DETERMIN FLOWDIRECTION (according to energy-potential)
                    best = 0
                    for r in range(-1,2):
                        for c in range(-1,2):
                            diff = E_water[i,j] - E_water[i+r,j+c]
                            if diff > 0:
                                if diff > best:
                                    best = diff
                                    ind_x = i+r
                                    ind_y = j+c
                                    if (r+c) == -2 or (r+c) == 2 or (r+c) == 0:
                                        width = np.sqrt(spatial_res**2 + spatial_res**2)
                                    else:
                                        width = spatial_res
                    if best == 0:
                        ind_x = i
                        ind_y = j
                        width = spatial_res

                    # If target pixxel not in catchment, pass
                    if catchment[ind_x,ind_y] == 0:
                        water[ind_x,ind_x] = 0
                        continue

                    ## SLOPE
                    s_water[i,j] = abs(E_water[i,j] - E_water[ind_x,ind_y]) / width

                    ## HYDRAULIC RADIUS
                    A_water = water[i,j] * width
                    P_water = 2 * water[i,j] + width
                    R_water = A_water / P_water

                    ## RUNOFF
                    mc_curr = mc_water[i,j] * (1 / mc_factor[t])
                    V_water[i,j] = 1 / mc_curr * R_water**(2/3) * s_water[i,j]**(1/2) # The higher the mc, the higher the friction
                    Q1 = V_water[i,j] * A_water # m^3 / s --> outflow of pixel at time t
                    Q_water[i,j] = Q1 / (width * width) * delta_t # m/h reduction

                    # SHIFTING
                    # Current Position
                    if Q_water[i,j] > water[i,j]:
                        Q_water[i,j] = water[i,j]
                    if Q_water[i,j] > max_shift:
                        Q_water[i,j] = max_shift
                    water[i,j] = water[i,j] - Q_water[i,j]
                    # Empty Borders
                    if i==0 or i==(x_max-x_min-1) or j==0 or j==(y_max-y_min-1):
                        water[i,j] = 0
                    # Target Pixxel
                    water_tmp[ind_x,ind_y] += Q_water[i,j]

                    #################################################################################
                    ## SOIL

                    ## DETERMIN FLOWDIRECTION
                    best = 0
                    for r in range(-1,2):
                        for c in range(-1,2):
                            diff = E_soil[i,j] - E_soil[i+r,j+c]
                            if diff > 0:
                                if diff > best:
                                    best = diff
                                    ind_x = i+r
                                    ind_y = j+c
                                    if (r+c) == -2 or (r+c) == 2 or (r+c) == 0:
                                        width = np.sqrt(spatial_res**2 + spatial_res**2)
                                    else:
                                        width = spatial_res
                    if best == 0:
                        ind_x = i
                        ind_y = j
                        width = spatial_res

                    # If target pixxel not in catchment, pass
                    if catchment[ind_x,ind_y] == 0:
                        soil[ind_x,ind_x] = 0
                        continue

                    ## SLOPE
                    s_soil[i,j] = abs(E_soil[i,j] - E_soil[ind_x,ind_y]) / width

                    ## HYDRAULIC RADIUS
                    A_soil = soil[i,j] * width
                    P_soil = 2 * soil[i,j] + width
                    R_soil = A_soil / P_soil

                    ## SUBSURFACE FLOW
                    V_soil[i,j] = 1 / mc_soil[i,j] * R_soil**(2/3) * s_soil[i,j]**(1/2)
                    Q1 = V_soil[i,j] * A_soil # m^3 / s --> outflow of pixel at time t
                    Q_soil[i,j] = Q1 / (width * width) * delta_t # m/h reduction

                    # SHIFTING
                    # Current Position
                    if Q_soil[i,j] > soil[i,j]:
                        Q_soil[i,j] = soil[i,j]
                    soil[i,j] = soil[i,j] - Q_soil[i,j]
                    # Borders
                    if i==0 or i==(x_max-x_min-1) or j==0 or j==(y_max-y_min-1):
                        soil[i,j] = 0
                    # Target Pixxel
                    soil_tmp[ind_x,ind_y] += Q_soil[i,j]

##########################################################################################

            ## UPDATES
            for i in range(size_x):
                for j in range(size_y):
                    CN[i,j] = (100 - CN_init[i,j]) / Sabs[i,j] * soil[i,j] + CN_init[i,j]
                    if CN[i,j] > 100:
                        CN[i,j] = 100
                    S[i,j] = (2540 / CN[i,j] - 25.4) / 1000
                    I[i,j] = lam[i,j] * S[i,j]

            # Update Water matrix
            water = water + water_tmp
            # Update Soil matrix
            soil = soil + soil_tmp
            # Overlandwater in the whole System
            water_latest = np.sum(water)
            soil_latest = np.sum(soil)
            
            # Store water amount
            water_storage[t+1,:,:] = water

            # Store water amount
            soil_storage[t+1,:,:] = soil
            
        print('DONDE')

        return water_storage, soil_storage, Q_water, Q_soil, water_tot, soil_tot, CN, V_water, V_soil, S


#_______________________________________________________________________________ MODEL POST-PROCESS


#------------------------------------------------------------------------------- MAIN RUN
if __name__ == '__main__':
    print('pomm main run')