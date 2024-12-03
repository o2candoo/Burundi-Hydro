from ursina import *
import random

# Ursina-Anwendung starten
app = Ursina()

# Interaktive Kamera aktivieren
EditorCamera()

# Anzahl der Punkte
num_dots = 100

# Liste, um die Punkte zu speichern
dots = []

# Punkte im 3D-Raum erzeugen und zufällig platzieren
for _ in range(num_dots):
    dot = Entity(
        model='sphere',  # Punkt-Model
        color=color.random_color(),  # Zufällige Farbe
        scale=0.05,  # Größe des Punktes
        position=(
            random.uniform(-4, 4),  # Zufällige x-Koordinate
            random.uniform(-4, 4),  # Zufällige y-Koordinate
            random.uniform(-4, 4)   # Zufällige z-Koordinate
        )
    )
    dots.append(dot)

# Textanzeige hinzufügen
info_text = Text(
    text=f'{num_dots} dots displayed in 3D!',
    position=(-0.7, 0.4),
    origin=(0, 0),
    color=color.white
)

# Hauptlauf der Anwendung
app.run()
