import ursina as urs
import numpy as np
import logging
from numba import njit

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


logging.basicConfig(level=logging.DEBUG)

## CONVERT GIS-IMAGE TO DOT-LIST
dots = generate_dots()


## INIT APP
app = urs.Ursina()
urs.window.borderless = False
# urs.window.size = (700, 700)
urs.window.color = urs.color.black

size = len(dots)
dots = np.array(dots)
print(dots.shape)
dots_2 = dots[[i for i in range(0,size,4)]]
dots_4 = dots[[i for i in range(0,size,8)]]
dots_8 = dots[[i for i in range(0,size,16)]]
terrains_dots = []
terrains_dots.append(urs.Entity(model=urs.Mesh(vertices=dots, mode='point', thickness=5), color=urs.color.rgba(0, 255, 0, 255)))
terrains_dots.append(urs.Entity(model=urs.Mesh(vertices=dots_2, mode='point', thickness=5), color=urs.color.rgba(255, 255, 255, 0)))
terrains_dots.append(urs.Entity(model=urs.Mesh(vertices=dots_4, mode='point', thickness=5), color=urs.color.rgba(255, 255, 255, 0)))
terrains_dots.append(urs.Entity(model=urs.Mesh(vertices=dots_8, mode='point', thickness=5), color=urs.color.rgba(255, 255, 255, 0)))


## INIT CAMERA
# urs.EditorCamera(rotation_smoothing=10, enabled=1, rotation=(30,30,0))
app.run()