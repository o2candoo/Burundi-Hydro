import ursina as urs
from numba import njit
import numpy as np
import os
import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


@njit(fastmath=True)
def generate_dots(img=None):
    dots = []
    if not img is None:
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                x = (i - (img.shape[0] // 2))
                y = (j - (img.shape[1] // 2))
                z = img[i,j] * -0.1
                if img[i,j] > 0:
                    dots.append((x, y, z))
    else:
        for x in range(10):
            for y in range(10):
                for z in range(10):
                    dots.append((x, y, z))
    return dots


def terrain_dots(img=None):
    """
    3D TERRAIN


    INPUT:
        - If None, then a standart 3d cube will be displayed to see if function is working


    OPTIONAL:
        1. GIS Image       [numerical numpy matrice]


    OUTPUT:
        - Ursina 3D app


    RETURN:
        - App and Entity for later updates
    """


    ## CONVERT GIS-IMAGE TO DOT-LIST
    dots = generate_dots(img)


    ## INIT APP
    app = urs.Ursina()
    urs.window.borderless = False
    urs.window.size = (700, 700)
    urs.window.color = urs.color.black

    size = len(dots)
    dots = np.array(dots)
    print(dots.shape)
    dots_2 = dots[[i for i in range(0,size,4)]]
    dots_4 = dots[[i for i in range(0,size,8)]]
    dots_8 = dots[[i for i in range(0,size,16)]]
    terrains_dots = []
    terrains_dots.append(urs.Entity(model=urs.Mesh(vertices=dots, mode='point', thickness=0.01), color=urs.color.rgba(0, 255, 0, 255)))
    terrains_dots.append(urs.Entity(model=urs.Mesh(vertices=dots_2, mode='point', thickness=0.01), color=urs.color.rgba(255, 255, 255, 0)))
    terrains_dots.append(urs.Entity(model=urs.Mesh(vertices=dots_4, mode='point', thickness=0.01), color=urs.color.rgba(255, 255, 255, 0)))
    terrains_dots.append(urs.Entity(model=urs.Mesh(vertices=dots_8, mode='point', thickness=0.01), color=urs.color.rgba(255, 255, 255, 0)))
    

    ## INIT CAMERA
    urs.EditorCamera(rotation_smoothing=10, enabled=1, rotation=(30,30,0))


    return app, terrains_dots


def terrain_mesh():
    """
    """


    # INITIALIZE SCENE
    app = urs.Ursina(vsync=False)
    urs.window.borderless = False
    elevation = load_data()


    # CHECK FOR TEXTURES
    textures = 0
    for file in os.listdir(os.path.dirname(os.path.realpath(__file__))):
        if 'texture' in file:
            textures += 1


    ## CREATE TERRAIN-ENTITIES WITH DIFFERENT TEXTURES
    terrains = []
    for i in range(textures):
        mountain = urs.Terrain(height_values=list(np.rot90(elevation, k=-1)), skip=1)
        terrain = urs.Entity(
            model=mountain,
            scale=(30,1,30),
            texture='./tmp_texture' + str(i+1),
            color=urs.color.rgba(0, 255, 0, 0),
        )
        terrains.append(terrain)


    ## ACTIVATE FIRST TERRAIN
    terrains[0].color = urs.color.rgba(0, 255, 0, 255)
    terrains[0].collider = 'mesh'
    terrains[0].collision = False


    ## CREATE 'HIGHLIGHT' ENTITY
    highlighted_terrain = urs.Entity(model='plane', scale=.1, texture='brick', visible=True)

    ## CAMERA
    urs.EditorCamera()


    return app, terrains, textures, highlighted_terrain


def load_data():
    file = '\\'.join(os.path.dirname(os.path.abspath(__file__)).split('\\')[:-1]) + '\\tmp\\tmp.npy'
    data = np.load(file)
    return data


def plot_system_water(tensor):
    """
    """
    ###
    print(tensor)
    print(tensor.shape)
    print(type(tensor))
    # Interpret image data as row-major instead of col-major
    pg.setConfigOptions(imageAxisOrder='row-major')
    app = QtWidgets.QApplication([])

    ## Create window with ImageView widget
    win = QtWidgets.QMainWindow()
    win.resize(800,800)
    imv = pg.ImageView()
    win.setCentralWidget(imv)
    win.show()
    win.setWindowTitle('pyqtgraph example: ImageView')

    lbl = QtWidgets.QLabel(win)
    lbl.setText('LABEL EXAMPLE')
    lbl.move(100,100)

    ## Display the data and assign each frame a time value
    # imv.setImage(np.sqrt(np.sqrt(p.storage)))
    print(np.max(tensor))
    print(np.min(tensor))
    imv.setImage(tensor)

    ## Set a custom color map
    colors = [
        (0, 0, 0),
        (125, 216, 167),
        (255, 255, 255)
    ]
    # cmap = pg.ColorMap(pos=np.linspace(0.0, 0.1, 3), color=colors)
    cmap = pg.ColorMap(pos=np.array([0.0,0.5,1.0]), color=colors)
    imv.setColorMap(cmap)

    ## Start Qt event loop unless running in interactive mode.
    # if (sys.flags.interactive != 1) or not ha ttr(QtCore, 'PYQT_VERSION'):
    QtWidgets.QApplication.instance().exec_()


if __name__ == '__main__':

    
    ## CHECK COMMANDLINE INPUT
    commands = sys.argv
    # commands.append('urs_terrain_mesh')

    if len(commands) == 1:
        commands.append('urs_cube_dots')
    if len(commands) > 1:


        ## URSINA
        if commands[1][:3] == 'urs':


            ## DOTS
            if commands[1][-4:] == 'dots':
                ## TERRAIN_DOTS
                if commands[1] == 'urs_terrain_dots':
                    app, terrains_dots = terrain_dots(img=load_data())
                ## STD CUBE
                elif commands[1] == 'urs_cube_dots':
                    app, terrains_dots = terrain_dots()
                ## USER INPUT FOR OBJECT ROTATION
                def input(key):
                    for t in terrains_dots:
                        if key == 'a': t.rotation_z += 10
                        if key == 'd': t.rotation_z -= 10
                        if key == 'w': t.rotation_x += 10
                        if key == 's': t.rotation_x -= 10
                        if key == 'q': t.rotation_y += 10
                        if key == 'e': t.rotation_y -= 10
                        if key == '1':
                            terrains_dots[0].color = urs.color.rgba(0, 255, 0, 255)
                            terrains_dots[1].color = urs.color.rgba(0, 255, 0, 0)
                            terrains_dots[2].color = urs.color.rgba(0, 255, 0, 0)
                            terrains_dots[3].color = urs.color.rgba(0, 255, 0, 0)
                        if key == '2':
                            terrains_dots[0].color = urs.color.rgba(0, 255, 0, 0)
                            terrains_dots[1].color = urs.color.rgba(0, 255, 0, 255)
                            terrains_dots[2].color = urs.color.rgba(0, 255, 0, 0)
                            terrains_dots[3].color = urs.color.rgba(0, 255, 0, 0)
                        if key == '3':
                            terrains_dots[0].color = urs.color.rgba(0, 255, 0, 0)
                            terrains_dots[1].color = urs.color.rgba(0, 255, 0, 0)
                            terrains_dots[2].color = urs.color.rgba(0, 255, 0, 255)
                            terrains_dots[3].color = urs.color.rgba(0, 255, 0, 0)
                        if key == '4':
                            terrains_dots[0].color = urs.color.rgba(0, 255, 0, 0)
                            terrains_dots[1].color = urs.color.rgba(0, 255, 0, 0)
                            terrains_dots[2].color = urs.color.rgba(0, 255, 0, 0)
                            terrains_dots[3].color = urs.color.rgba(0, 255, 0, 255)


            ## MESH
            if commands[1][-4:] == 'mesh':
                ## INITIALIZE SCENE
                app, terrains, textures, highlighted_terrain = terrain_mesh()
                app.active = 1
                ## UPDATES
                def update():
                    try:
                        tooltip_test.text = urs.mouse.world_point.y
                        highlighted_terrain.position = urs.mouse.world_point
                        # highlighted_terrain.rotation = (urs.mouse.world_normal.x, urs.mouse.world_normal.y, urs.mouse.world_normal.z)
                        highlighted_terrain.rotation = urs.terrains[0].model.normals
                    except:
                        pass
                ## MOUSE / KEYBOARD INPUT
                def input(key):
                    ## CHECK ACTIVE TEXTURE
                    if key == 'right arrow':
                        if app.active == textures:
                            app.active = 1
                        else:
                            app.active += 1
                    if key == 'left arrow':
                        if app.active == 1:
                            app.active = textures
                        else:
                            app.active -= 1
                    ## UPDATE ALL TERRAINS
                    for i, terrain in enumerate(terrains):
                        if key == 'up arrow':
                            terrain.scale = (terrain.scale[0], terrain.scale[1]+1, terrain.scale[2])
                        if key == 'down arrow':
                            terrain.scale = (terrain.scale[0], terrain.scale[1]-1, terrain.scale[2])
                        ## SET TEXTURE
                        if i == (app.active-1):
                            terrains[i].color = urs.color.rgba(0, 255, 0, 255)
                        else:
                            terrains[i].color = urs.color.rgba(0, 255, 0, 0)
                    ## ENABLE/DISABLE TOOLTIP
                    if key == 't':
                        if tooltip_test.enabled == True:
                            tooltip_test.enabled = False
                        else:
                            tooltip_test.enabled = True
                    ## ENABLE/DISABLE MESH
                    if key == 'm':
                        if terrains[0].collision == False:
                            terrains[0].collision = True
                        else:
                            terrains[0].collision = False
                        
                ## BUTTON
                def Click():
                    txt = urs.Text(text=str(app.active),scale = 1,position = (-0.75,-0.45,0))
                    urs.destroy(txt,delay=2)
                Option = urs.Button(model="circle",texture = 'brick',color = urs.color.blue,scale = 0.05, position=(-0.85,-0.45,0), highlight_color=urs.color.yellow)
                Option.on_click=Click
                ## TOOLTIP
                tooltip_test = urs.Tooltip(
                    '<scale:1.5><pink>' + 'Rainstorm' + '<scale:1> \n \n' +
                    '''Summon a <blue>rain
                    storm <default>to deal 5 <blue>water
                    damage <default>to <red>everyone, <default>including <orange>yourself. <default>
                    Lasts for 4 rounds.'''.replace('\n', ' '),
                )
                tooltip_test.enabled = False


            ## APP
            app.run()


        ## TENSOR PLOT IN QT
        elif commands[1] == 'qt_giv':
            plot_system_water(tensor=load_data())
        elif commands[1] == 'qt_std':
            pass
    else:
        pass
    
    print('End')