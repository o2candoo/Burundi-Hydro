#------------------------------------------------------------------------------- SCRIPT INFO
'''
File name:              helper.py
Author:                 Oliver Carmignani
Date of creation:       02/28/2023
Date last modified:     02/28/2023
Python Version:         3.10
'''


#------------------------------------------------------------------------------- MODULES
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import os
import plotly.express as px
from skimage import io
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import statsmodels.api as sm
import pandas as pd


#------------------------------------------------------------------------------- SETTINGS
## IGNORE WARNINGS IN JUPYTER NOTEBOOK
warnings.filterwarnings('ignore')
## APPEARENCE
color_pal = sns.color_palette()
plt.style.use('ggplot')


#------------------------------------------------------------------------------- FUNCTIONS
## PRINT MAP
def print_map(df_loc, zoom=10):
    '''
    Display location on google street map in plotly

    INPUT:
        - df_loc                Dataframe with locations        [pandas]
        - displayname2compID    Dictionary with name/id         [dict]

    OUTPUT:
        - plotly scatterplot

    RETURN:
        - None
    '''

    fig = px.scatter_mapbox(
        df_loc,
        lat=df_loc['latitude'],
        lon=df_loc['longitude'],
        zoom=zoom,
        # color=df_loc['Group'],
        # hover_data={'displayname':df_loc['displayname']},
        width=800,
        height=600
    )
    fig.update_layout(mapbox_style='open-street-map')
    fig.update_layout(margin={'r':0, 't':50, 'l':0, 'b':10})
    fig.show()


## FEATURE TARGET
def feature_target_relation(df, key, ylabel='', feature='hour', location='test', name='', vis=True, subdir=''):
    '''
    Boxplot overview of a Timeseries of a Pandas-DataFrame with subselection of timefeature

    INPUT:
        - df            pandas Dataframe                        [pandas DataFrame]
        - key           key of pandas Dataframe to visualize    [string]
        - ylabel        ylabel of the plot                      [string]
        - feature       timefeature to examine                  [string: 'minute', 'hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']
        - plant         plant name                              [string]
        - name          name of the component                   [string]
        - vis           Visualize the plot                      [boolean]
        - subdir        name of subdir for saving the plot      [string]

    OUTPUT:
        - Boxplot

    RETURN:
        - Saved Boxplot in output/<plantname>/<subdir>/<title>.png
    '''
    ## CHECK IF TIMEFEATURES ALREADY EXIST
    try:
        testvariable = df['hour']
    except Exception as e:
        print(e)
        print('POMM ERROR: Input has no timefeatures. Try to run function "create_timefeatures(<pandasdataframe>)"')

    ## PLOT SETTINGS
    title = location + ' -- ' + name + ' -- ' + key + ' by ' + feature
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=df, x=feature, y=key)
    ax.set_title(title)
    ax.set_ylabel(ylabel)

    ## SAVE PLOT
    mydir = './output/plots/' + subdir
    name = mydir + title + '.png'
    if not os.path.isdir(mydir):
        print(f'Creating folder {mydir}')
        os.mkdir(mydir)
    plt.savefig(name)

    ## SHOW PLOT
    if vis:
        plt.show()
    else:
        plt.close()


## PLOT TS OVERVIEW
def plot_overview(df, key, title, location='test', ylabel='', alpha=0.05, width=20, height=5, ylim=None, outliers=None, vis=True, subdir=''):
    '''
    Scatterplot overview of a Timeseries of a Pandas-DataFrame

    INPUT:
        - df            pandas Dataframe                        [pandas DataFrame]
        - key           key of pandas Dataframe to visualize    [string]
        - title         title of the plot                       [string]
        - plant         plant name for output-file-name         [string]
        - ylabel        ylabel of the plot                      [string]
        - alpha         transparency of the dots                [float]
        - width         plot width                              [int]
        - height        plot height                             [int]
        - ylim          y axis limitations                      [int]
        - outliers      mark outliers                           [boolean Vector according to pandas DataFrame]
        - vis           Visualize the plot                      [boolean]
        - subdir        name of subdir for saving the plot      [string]

    OUTPUT:
        - Scatterplot

    RETURN:
        - Saved scatterplot in output/<plantname>/<subdir>/<title>.png
    '''
    ## PLOT SETTINGS
    fig, ax = plt.subplots(figsize=(width,height))
    if outliers is None:
        ax.plot(df.index, df[key], '-o', alpha = alpha, label=key, color='blue')
        ax.plot(df.index, df[key], label=key, color='red')
    else:
        ax.plot(df.index[-outliers], df[key][-outliers], 'o', alpha=alpha, label=key, color='blue')
        ax.plot(df.index[-outliers], df[key][-outliers], label=key, color='red')
        ax.plot(df.index[outliers], df[key][outliers], 'o', label='Outsiders', color='red')


    ## LABELS AND LEGEND
    fig.autofmt_xdate()
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if not ylim is None:
        ax.set_ylim(ylim)
    plt.legend()


    ## SAVE PLOT
    mydir = './output/plots/' + subdir
    name = mydir + title + '.png'
    if not os.path.isdir(mydir):
        print(f'Creating folder {mydir}')
        os.mkdir(mydir)
    plt.savefig(name)


    ## SHOW PLOT
    if vis:
        plt.show()
    else:
        plt.close()


## PLOT TIF OVERVIEW
def im_show(img, title=('Test',), s_shape=[1,1], cmap=('viridis',), bbox=[0,0,0,0], bbox_on=(True,), mode=('img',), name=None, vis=True, norm=False):
    """
    GIS IMAGE
        Displays GIS images (tif), given as numpy matrices.
        Boundingbox optional to turn on for each plot separately.

    INPUT:
        1. img          GIS Images          [Tuple --> numerical numpy matrices]
    
    OPTIONAL:
        2. title        Titles              [Tuple --> strings]
        3. s_shape      Placement           [List --> horizontal and vertical amount of images]
        4. cmap         Colormap            [Tuple --> colormaps] --> https://matplotlib.org/stable/tutorials/colors/colormaps.html
        5. bbox         Boundingboxes       [List --> xmin, xmax, ymin, ymax]
        6. bbox_on      Enable Bboxes       [Tuple --> integers --> 1=On, 0=Off]
        7. mode         Display mode        [Tuple --> strings --> img;hist;ord]
        8. name         Export              [String --> Name of outputfile including extension. 'Test.png']
        9. vis          Display             [Boolean]
        10. norm        Normalize           [Boolean]

    OUTPUT:
        - Matplotlib Image Plot

    RETURN:
        None
    """


    ## CHECK INPUT
    if type(img) != tuple:
        img = (img,)
    if not len(title) == len(img):
        title = [title[0]] * len(img)
    if not s_shape[0] * s_shape[1] == len(img):
        s_shape[0] = len(img)
    if not len(cmap) == len(img):
        cmap = [cmap[0]] * len(img)
    if not len(bbox_on) == len(img):
        bbox_on = [bbox_on[0]] * len(img)
    if not len(mode) == len(img):
        mode = [mode[0]] * len(img)

    
    ## INITIALIZE PLOT
    fig = plt.figure(figsize=(20,10))
    ax = [0] * len(img)


    ## SUBPLOTS
    for i in range(len(img)):


        ## SUBPLOT
        ax[i] = plt.subplot(s_shape[0], s_shape[1], i+1)
        ax[i].set_title(title[i])
        if mode[i] == 'img':
            im1 = ax[i].imshow(img[i], cmap=plt.get_cmap(cmap[i]))
            ## COLORBAR
            divider = make_axes_locatable(ax[i])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im1, cax=cax, orientation='vertical')
        elif mode[i] == 'ord':
            tmp = np.copy(img[i]).astype(int).ravel()
            tmp = tmp[np.where(tmp>0)]
            if norm:
                ax[i].plot(np.sort(tmp/np.max(tmp), axis=None))
            else:
                ax[i].plot(np.sort(tmp, axis=None))
        elif mode[i] == 'hist':
            tmp = np.copy(img[i]).astype(int).ravel()
            tmp = tmp[np.where(tmp>0)]
            bin_steps = np.max(tmp) // 15
            # bins = [int(i*bin_steps) for i in range(16)]
            bins = [i for i in range(0, np.max(tmp), 100)]
            if norm:
                ax[i].hist(tmp, bins, density=True)
            else:
                ax[i].hist(tmp, bins)
        else:
            print('POMM ERROR: Mode not valid.')
        ax[i].grid(False)


        ## BOUNDING BOX
        if bbox_on[i]:
            rect = patches.Rectangle((bbox[2], bbox[0]), bbox[3]-bbox[2], bbox[1]-bbox[0], linewidth=1, edgecolor='r', facecolor='none')
            ax[i].add_patch(rect)


    plt.tight_layout()

    if not name is None:
        plt.savefig('./output/plots/' + name + '.png', dpi=300, bbox_inches='tight')
    if vis:
        plt.show()


## PLOT TENSOR (THREAD)
def plot_tensor(tensor=None):
        """
        SYSTEM WATER
            Displays a qt img viewer over time fop given numpy tensor.
            Starts an external python script so the QT-app can be closed and reopened properely, if necessary.

        INPUT:
            - If None, then ...

        OPTIONAL:
            1. tensor           Image Array       [numerical numpy tensor]

        OUTPUT:
            - QT app

        RETURN:
            - None
        """


        ## RUN EXTERNAL PYTHON FILE FOR GIVEN TENSOR, ELSE RUN FOR STD TENSOR
        path = '\\'.join(os.path.dirname(os.path.abspath(__file__)).split('\\')[:-1])
        if not tensor is None:
            np.save(path + '\\pomm\\tmp\\tmp.npy', tensor)
            os.system('python ' + '"' + path + '\\pomm\\tmp\\tmp.py" qt_giv')
        else:
            os.system('python ' + '"' + path + '\\pomm\\tmp\\tmp.py" qt_std')


## PLOT 3D SURFACE (THREAD)
def terrain_3d(elevation=None, textures=None, mode='dots'):
    """
    3D TERRAIN
        Displays a dotcloud in a 3D space with library 'Ursina'.
        Starts an external python script so the Ursina-app can be closed and reopened properely, if necessary.

    INPUT:
        - If None, a standart 3D dotcloud-cube will be displayed.

    OPTIONAL:
        1. img         GIS Image       [numerical 3D numpy matrice]

    OUTPUT:
        - Ursina 3D app

    RETURN:
        - None
    """
    ## DELETE PRE-EXISTING-FILES
    for file in os.listdir(os.path.dirname(os.path.realpath(__file__)) + '\\tmp'):
        if 'texture' in file:
            os.remove(os.path.dirname(os.path.realpath(__file__)) + '\\tmp\\' + file)

    ## SAVE TEMPORARY FILES
    path = '\\'.join(os.path.dirname(os.path.abspath(__file__)).split('\\')[:-1])
    if not elevation is None:
        if textures is None:
            textures = elevation.copy()
        if not type(textures)==list: textures = [texture]
        # ELEVATION
        ## SET NAN TO 0, THEN SET 0 TO 1E5, SHIFT ELEVATION MIN TO 0, AND SET ALL ABOVE 1E4 TO 0
        elevation = np.nan_to_num(elevation)
        elevation[elevation==0] = 1e5
        elevation -= elevation.min()
        elevation[elevation>1e4] = 0
        np.save(path + '\\pomm\\tmp\\tmp.npy', elevation)
        # TEXTURES
        for i, texture in enumerate(textures):
            ## SET NAN TO 0, THEN SET 0 TO 1E5, SHIFT TEXTURE MIN TO 0, AND SET ALL ABOVE 1E4 TO 0
            texture = np.nan_to_num(texture)
            texture[texture==0] = 1e5
            texture -= texture.min()
            texture[texture>1e4] = 0
            ## SET TEXTURE MIN TO 0, NORMALIZE TO [0, 255]
            texture = texture / texture.max()
            texture *= -255
            texture += 255
            io.imsave(path + '\\pomm\\tmp\\tmp_texture' + str(i+1) + '.png', texture.astype(np.uint8))


    ## RUN EXTERNAL PYTHON FILE
    if mode=='dots':
        if not elevation is None:
            os.system('python ' + '"' + path + '\\pomm\\tmp\\tmp.py" urs_terrain_dots')
        else:
            os.system('python ' + '"' + path + '\\pomm\\tmp\\tmp.py" urs_cube_dots')
    elif mode=='mesh':
        if not elevation is None:
            os.system('python ' + '"' + path + '\\pomm\\tmp\\tmp.py" urs_terrain_mesh')


## PLOT RAINFALL RUNOFF
def plot_rainfall_runoff(df, rain, runoff, model=None, startdate=None, days=None, plot_title='Rainfall - Runoff'):
    """
    RAINFALL - RUNOFF
        Displays a top-down RainfallRunoff Timeseries.
        Brakets-Window optional for sub-interval.
    
    INPUT:
        1. rain             Rainfall Timeseries         [numerical vector]
        2. runoff           Runoff Timeseries           [numerical vector]

    OPTIONAL:
        3. model            Model Timeseries            [numerical vector]
        4. startdate        YYYY-MM-DD                  [string]
        5. plot_title                                   [string]
        6. T_start          Brakets Start               [numerical integer]
        7. T_end            Brakets End                 [numerical integer]

    OUTPUT:
        - Matplotlib Timeseries Plot

    RETURN:
        None
    """

    ## INIT PLOT
    fig, ax1 = plt.subplots(figsize=(20,5))
    fig.autofmt_xdate()
    x = np.linspace(0, len(df[runoff]), len(df[runoff]))

    ## PLOT RUNOFF
    ax1.grid()
    ax1.set_title(plot_title)
    ax1.plot(df.index, df[runoff], color='red', label='Validation')
    if not model is None:
        ax1.plot(df.index, df[model], color='green', label='Model')
    ax1.set(ylim=(0, np.max(df[runoff])*1.5))
    ax1.set_ylabel('Streamflow [$m^3$/s]')
    ax1.set_xlabel('Day')

    ## PLOT RAINFALL
    ax2 = ax1.twinx()
    ax2.plot(df.index, df[rain], color='blue', label='Rainfall')
    ax2.fill_between(df.index, 0, df[rain], alpha=0.8, color='blue')
    ax2.set(ylim=(0, np.max(df[rain])*3))
    ax2.set_ylabel('Rainfall [mm]')
    ax2.invert_yaxis()

    # ## ADD START VERTICAL LINE
    if not startdate is None and startdate in df.index:
        ax2.axvline(df.index[df.index==startdate],0,1, color='black')
    else:
        ax2.axvline(df.index[0],0,1, color='black')
        left, bottom, width, height = (mdates.date2num(df.index[0]), 0, df.shape[0], np.max(df[runoff])*1.5)
        rect=patches.Rectangle((left,bottom),width,height, 
                                fill=True,
                                alpha=0.3,
                                color="green",
                                linewidth=2)
    print(rect)
    ax1.add_patch(rect)

    ## ADD LEGEND
    ax1.legend(loc='center right')
    ax2.legend(loc='center left')

    ## SAVE PLOT
    mydir = './output/plots/'
    name = mydir + plot_title + '.png'
    if not os.path.isdir(mydir):
        print(f'Creating folder {mydir}')
        os.mkdir(mydir)
    plt.savefig(name)

    ## SHOW PLOT
    plt.show()


## SHOW MANNINGS-COEFFICIENT WITH CURVE NUMBER
def plot_mc_cn(ts=[], CN_borders=[], vis=True):
    """
    """
    ## Visualize
    cn = np.array([0.55, 0.6, 0.66, 0.66, 0.85, 0.6, 0.86, 0.76])
    mc = np.array([0.6, 0.6, 0.5, 0.45, 0.012, 0.45, 0.018, 0.02])
    landuse = ['Natural forest', 'Secondary forest', 'Mixed agriculture', 'Plantation', 'Settlements', 'Shrub', 'Bareland', 'Dryland farming']
    pos_mc = np.array([-0.025, 0, 0, 0, +0.015, 0, -0.015, +0.015])
    pos_cn = np.array([-0.025, 0.005, 0.005, 0.005, -0.015, 0.005, 0.005, -0.015])
    plt.scatter(cn, mc)
    plt.xlim((0.5, 1.0))
    plt.ylim((0.0, 0.8))
    for i, txt in enumerate(landuse):
        plt.annotate(txt, (cn[i] + pos_cn[i], mc[i] + pos_mc[i]))
    if ts != [] and CN_borders!= []:
        ## Add dist of current catchment
        cn_run = np.array([CN_borders[1]/100, CN_borders[2]/100, CN_borders[3]/100, CN_borders[4]/100, CN_borders[5]/100])
        mc_run = np.array([ts[CN_borders[1]] / 1000, ts[CN_borders[2]] / 1000, ts[CN_borders[3]] / 1000, ts[CN_borders[4]] / 1000, ts[CN_borders[5]] / 1000])
        s = np.array([10,10,10,10,10])
        plt.scatter(cn_run, mc_run, s=s)
    if vis==True:
        plt.xlabel('Curve Number (CN)')
        plt.ylabel("Manning's coefficient (MC)")
        plt.title('MC - CN - Distribution')
        plt.show()


## PLOT 3D SURFACE IN MATPLOTLIB
def plot_3d(x, y, z):
        """
        """
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(x,y,z,rstride=5,cstride=5,alpha=0.3,cmap=plt.cm.coolwarm)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('$m$')
        # # Slider
        # axcolor = 'lightgoldenrodyellow'
        # factor_freq = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
        # sexp = Slider(factor_freq, 'Freq', 0, T, valinit=0, valstep=1)
        # # Water Field Function
        # def update(val):
        #     lv = sexp.val
        #     # Water Field
        #     ax.clear()
        #     ax.plot_surface(x,y,z,rstride=5,cstride=5,alpha=0.3,cmap=plt.cm.coolwarm)
        # sexp.on_changed(update)
        plt.show()


## PLOT NETCDF-DISTRIBUTION
def plot_nc(ds, lat_size, lon_size):
    """
    """
    fig, ax = plt.subplots(1,1)
    img = ax.imshow(np.array(ds.variables['skt'])[0])
    ax.set_xticks(range(lon_size))
    ax.set_xticklabels(np.array(ds.variables['longitude']))
    ax.set_yticks(range(lat_size))
    ax.set_yticklabels(np.array(ds.variables['latitude']))
    fig.colorbar(img)


## PLOT FLOW DURATION CURVE WITH 
def plot_fdc(df, key=False, ylabel='Runoff [m^3/s]', log=False, title='FDC'):
    """
    """
    plt.figure(figsize=(15,5))
    df = pd.DataFrame(df)
    df = df.dropna()
    perc = np.linspace(0,100,len(df))
    if log:
        plt.yscale('log')
    if key:
        y = np.array(df.sort_values(by=[key], ascending=False)[key])
        Q10 = np.quantile(y, 0.9)
        Q30 = np.quantile(y, 0.7)
        Q50 = np.quantile(y, 0.5)
        Q70 = np.quantile(y, 0.3)
        Q95 = np.quantile(y, 0.05)
        plt.plot(perc, y, label=key, color=(125/255, 216/255, 167/255))
        plt.axvline(10, color='black', alpha=0.2)
        plt.text(10-2, np.max(y)/2, f'Q10: {round(Q10, 2)}', fontsize=12, rotation=90)
        plt.axvline(30, color='black', alpha=0.2)
        plt.text(30-2, np.max(y)/2, f'Q30: {round(Q30, 2)}', fontsize=12, rotation=90)
        plt.axvline(50, color='black', alpha=0.2)
        plt.text(50-2, np.max(y)/2, f'Q50: {round(Q50, 2)}', fontsize=12, rotation=90)
        plt.axvline(70, color='black', alpha=0.2)
        plt.text(70-2, np.max(y)/2, f'Q70: {round(Q70, 2)}', fontsize=12, rotation=90)
        plt.axvline(95, color='black', alpha=0.2)
        plt.text(95-2, np.max(y)/2, f'Q95: {round(Q95, 2)}', fontsize=12, rotation=90)
    else:
        colors = ['#d62728', '#1f77b4', '#9467bd', '#7f7f7f', '#ff7f0e', '#bcbd22']
        for i, key in enumerate(df.keys()):
            y = np.array(df.sort_values(by=[key], ascending=False)[key])
            Q10 = np.quantile(y, 0.9)
            Q30 = np.quantile(y, 0.7)
            Q50 = np.quantile(y, 0.5)
            Q70 = np.quantile(y, 0.3)
            Q95 = np.quantile(y, 0.05)
            plt.plot(perc, y, label=key)
            plt.axvline(10, color='black', alpha=0.2)
            plt.text(10-2+i*2.5, np.max(y)/2, f'Q10: {round(Q10, 2)}', fontsize=12, rotation=90, color=colors[i])
            plt.axvline(30, color='black', alpha=0.2)
            plt.text(30-2+i*2.5, np.max(y)/2, f'Q30: {round(Q30, 2)}', fontsize=12, rotation=90, color=colors[i])
            plt.axvline(50, color='black', alpha=0.2)
            plt.text(50-2+i*2.5, np.max(y)/2, f'Q50: {round(Q50, 2)}', fontsize=12, rotation=90, color=colors[i])
            plt.axvline(70, color='black', alpha=0.2)
            plt.text(70-2+i*2.5, np.max(y)/2, f'Q70: {round(Q70, 2)}', fontsize=12, rotation=90, color=colors[i])
            plt.axvline(95, color='black', alpha=0.2)
            plt.text(95-2+i*2.5, np.max(y)/2, f'Q95: {round(Q95, 2)}', fontsize=12, rotation=90, color=colors[i])
    plt.legend()
    plt.title(title)
    plt.xlabel('Percentage [%]')
    plt.ylabel(ylabel)

    ## SAVE PLOT
    mydir = './output/plots/'
    name = mydir + title + '.png'
    if not os.path.isdir(mydir):
        print(f'Creating folder {mydir}')
        os.mkdir(mydir)
    plt.savefig(name)

## PLOT LOESS
def plot_loess(df, key, title, name=None):
    """
    """
    mean_average_annual_precip = np.mean(df[key].groupby(pd.Grouper(freq='1Y')).mean())
    annual_precip_means = df[key].groupby(pd.Grouper(freq='1Y')).mean()
    plt.figure(figsize=(10, 4))
    x = np.unique(df.year)
    y = (annual_precip_means-mean_average_annual_precip)/mean_average_annual_precip*100
    # Calculate the LOESS smoothing line
    lowess = sm.nonparametric.lowess(y, x, frac=0.3)
    smoothed_x, smoothed_y = lowess.T
    plt.bar(x, y, color='grey', alpha=0.5)
    plt.plot(smoothed_x, smoothed_y, label="LOESS Smoothing", color=(125/255, 216/255, 167/255), linewidth=2)
    plt.ylabel('Deviations in %')
    plt.title(title)

    if not name is None:
        plt.savefig('./output/plots/' + name + '_loess.png', dpi=300, bbox_inches='tight')

    plt.show()

#------------------------------------------------------------------------------- MAIN RUN
if __name__ == '__main__':
    print('pomm_vis main run')


    # # Surface Properties
    # size = 40       
    # # Example 1
    # xv = np.linspace(-5, 5, size)
    # yv = np.linspace(0, 5, size)
    # # # Example 2
    # # xv = np.linspace(0, 10, size)
    # # yv = np.linspace(0, 10, size)
    # # # Example 3
    # # xv = np.linspace(-4, 8, size)
    # # yv = np.linspace(-4, 8, size)
    # # Generate Surface
    # x,y = np.meshgrid(xv, yv)
    # z = (9-x**2 - y**2) * -10 + 100
    # plot_3d(x=x, y=y, z=z)

    # terrain_3d(elevation=z, type='dots')

    # terrain_3d(elevation=z, texture=z, type='mesh')