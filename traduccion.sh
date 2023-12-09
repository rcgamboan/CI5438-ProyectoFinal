#!/bin/bash

python3 src/limpieza.py spa 50000
python3 src/entrenamiento.py spa 50000
python3 src/traductor.py spa 50000