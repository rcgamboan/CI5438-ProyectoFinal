# CI5438-ProyectoFinal

## Descripción
Este proyecto contiene la implementación de un traductor basado en redes neuronales recurrentes (RNR). La implementación acepta traducciones del inglés al español, sin embargo se puede entrenar el modelo para la inclusión de más idiomas.

## Requisitos
- Python 3.6 o superior
- TensorFlow 2.13.0
- Sklearn 0.0.post5
- Numpy 1.23.4
- Keras 2.13.1

## Instalación
1. Clona este repositorio: `git clone https://github.com/tu-usuario/tu-proyecto.git`
2. Navega al directorio del proyecto: `cd ruta-al-repositorio`
3. Instala las dependencias: `pip install -r requirements.txt` (en caso de ser necesario)

## Ejecución del proyecto
### Ejecución completa

Para ejecutar el proyecto realizando el procedimiento de entrenamiento completo utilizar el siguiente comando:
```markdown
```bash
traduccion.sh
```
Este bash script contiene las comandos de linea para realizar una ejecución completa del proyecto, que va desde el preprocesamiento de los datos, entrenamiento de la red neuronal y la traducción de las oraciones que el cliente desee.  
En caso de no querer usar el bash script, se pueden ejecutar los archivos en el siguiente orden:  
```markdown
```bash
python3 limpieza.py target_lang size
python3 entrenamiento.py target_lang size
python3 traductor.py target_lang size
```
donde:  
target_lang: se refiere a las tres primeras letras del idioma al que se quiere traducir.  
size: pertenece a la cantidad de pares, tanto del idioma origen como del idioma destino, que se van a tomar para el entrenamiento de la red neuronal.

### Ejecución única del traductor
Luego de que ya se tiene el entrenamiento del modelo listo, puede ser ejecutado solamente el traductor, de la siguiente manera:
```markdown
```bash
python3 traductor.py target_lang size
```

## Estructura del Proyecto
- /src: carpeta con el código fuente del proyecto
- /data: conjunto de datos utilizados por el proyecto para el entrenamiento de la red neuronal
- /data/clean_data: conjunto de datos limpio para ser usado por la red neuronal
- /training_checkpoints: data referente al entrenamiento de la red
