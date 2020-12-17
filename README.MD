{\rtf1\ansi\ansicpg1252\cocoartf2513
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # [De]velar\
Este sitio busca ser una web experimental interactiva que, a trav\'e9s de la tecnolog\'eda de OpenCV, busque subvertir un algoritmo de detecci\'f3n facial, eliminando la fuente original de v\'eddeo; obteniendo as\'ed un nuevo material, compuesto \'fanicamente por las huellas de este algoritmo y la se\'f1al original de sonido del v\'eddeo. \
\
[De]velar deviene del trabajo colectivo del Lab. de Anarchivo organizado durante el \'faltimo trimestre de 2020 por Plataforma Bogot\'e1.\
\
# Prototipo 0.1 [actual]\
Por el momento el sitio cuenta con una peque\'f1a introducci\'f3n y con una interacci\'f3n con la c\'e1mara de v\'eddeo del usuari@. Dicha interacci\'f3n es luego pasada por el sample de face-detection de [Ningxin Hu](https://huningxin.github.io/opencv.js/samples/face-detection/index.html), eliminando la visibilidad del objeto html ```<video>``` en que se aloja el stream de la c\'e1mara.\
\
# Desaf\'edos\
A continuaci\'f3n una lista de lo que falta para continuar con el sitio:\
\
- Conectar un input simple de v\'eddeo y dise\'f1ar su interfaz.\
- Conectar el sample de opencv.js con el input.\
- Decidir si mostrar el procesamiento del v\'eddeo o hacer una barra de carga. Implementar.\
- Entregar un output del v\'eddeo procesado como elemento de v\'eddeo de HMTL5.}