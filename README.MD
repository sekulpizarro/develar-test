#[De]velar
Este sitio busca ser una web experimental interactiva que, a través de la tecnología de OpenCV, busque subvertir un algoritmo de detección facial, eliminando la fuente original de vídeo; obteniendo así un nuevo material, compuesto únicamente por las huellas de este algoritmo y la señal original de sonido del vídeo. 

[De]velar deviene del trabajo colectivo del Lab. de Anarchivo organizado durante el último trimestre de 2020 por Plataforma Bogotá.

#Prototipo 0.1 [actual]
Por el momento el sitio cuenta con una pequeña introducción y con una interacción con la cámara de vídeo del usuari@. Dicha interacción es luego pasada por el sample de face-detection de [Ningxin Hu](https://huningxin.github.io/opencv.js/samples/face-detection/index.html), eliminando la visibilidad del objeto html ```<video>``` en que se aloja el stream de la cámara.

#Desafíos
A continuación una lista de lo que falta para continuar con el sitio:

- Conectar un input simple de vídeo y diseñar su interfaz.
- Conectar el sample de opencv.js con el input.
- Decidir si mostrar el procesamiento del vídeo o hacer una barra de carga. Implementar.
- Entregar un output del vídeo procesado como elemento de vídeo de HMTL5.
