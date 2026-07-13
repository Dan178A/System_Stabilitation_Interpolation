


## UNIVERSIDAD DEL ZULIA
## FACULTAD EXPERIMENTAL DE CIENCIAS
## CONVENIO LUZ - IUTA
## PROYECTO DE TRABAJO DE INVESTIGACIÓN








Sistema de estabilización de imágenes en dispositivos móviles utilizando
interpolación y aprendizaje automático


Trabajo presentado como requisito para optar
al título de Licenciado en Computación




## Autor: Daniel Alejandro Silva Rojas

Tutor: María del Pilar López Molina







Guarenas, marzo de 2025












Sistema de estabilización de imágenes en dispositivos móviles utilizando
interpolación y aprendizaje automático





## Daniel Alejandro Silva Rojas
C.I. No.: 28.576.834
## Teléfono: +58 414 2317561
Urb. Altos de Copacabana, Guarenas
Correo electrónico: dsrglrm@gmail.com



María del Pilar López Molina
C.I. No.: 14.267.087
Correo electrónico: lopez.mapi@gmail.com




Universidad del Zulia
Facultad Experimental de Ciencias División
de Programas Especiales
Licenciatura en
## Computación


## VEREDICTO



Nosotros, los abajo firmantes, Profesor Gerardo Pirela (Coordinador), C.I. 12.404.565,
Profesor Alfredo Acurero, C.I. 9.783.996, y Profesora María del Pilar López, C.I.
14.267.087, designados por el Consejo Técnico en su sesión ordinaria Nº CTDPE-002-
2025 de fecha 21-04-2025, para evaluar el Trabajo de Investigación:

## “SISTEMA DE ESTABILIZACIÓN DE IMÁGENES EN DISPOSITIVOS MÓVILES
## UTILIZANDO INTERPOLACIÓN Y APRENDIZAJE AUTOMÁTICO”

Presentado por el TSU Daniel Alejandro Silva Rojas, titular de la Cédula de Identidad Nº
28.576.834, para optar al título de Licenciado en Computación, lo declaramos
APROBADO, obteniendo una calificación de veinte (20) puntos. Siendo su tutor la
Profesora María del Pilar López, titular de la Cédula de Identidad Nº 14.267.087.

Se le otorga la MENCIÓN PUBLICACIÓN, por su exploración de múltiples arquitecturas de
punta que resultó en una mezcla ingeniosa de técnicas de aprendizaje de máquina en
plataformas híbridas, bajo restricciones de infraestructura, pero sin comprometer el
desempeño de la solución computacional.

Maracaibo, a los veinticinco días del mes de abril de dos mil veinticinco.


## Prof. Alfredo Acurero Prof. Gerardo Pirela
## C.I. 9.783.996 C.I. 12.404.565
## Coordinador Miembro

Profa. María del Pilar López
## C.I. 14.267.087
## Miembro



Av. Universidad con prolongación Av. 5 de julio. Edificio Grano de Oro
## Telf.: 0261 412.77.35
www.fec.luz.edu.ve / Maracaibo Edo. Zulia




## DEDICATORIA
Dedico este trabajo a mis seres queridos que están conmigo en vida, por su
apoyo incondicional y amor constante. Agradezco profundamente a mi familia y
amigos por estar siempre a mi lado, brindándome fuerza y motivación.

Quiero conmemorar especialmente a mi mascota, compañero y mejor amigo,
Oddi. Aunque ya no estés físicamente conmigo, tu recuerdo y el amor que
compartimos siguen vivos en mi corazón. Gracias, Oddi, por todos los momentos de
alegría y compañía que me diste. Este trabajo es también para ti.





## AGRADECIMIENTO

Quiero expresar mi más profundo agradecimiento a todas las personas y
seres que han sido fundamentales en mi vida y en la realización de este trabajo. A
Dios, por guiarme y darme la sabiduría y fuerza para superar desafíos y alcanzar
mis metas. A mi familia, por su amor incondicional, apoyo constante y por ser mi
pilar en todo momento. Gracias por creer en mí y darme la fuerza para seguir
adelante.
A mis queridas mascotas, que llenaron mis días de alegría y ternura con su
compañía y amor. En este rincón especial de mi corazón, Oddi, mi fiel compañero y
mejor amigo, se alza como un faro de luz. Con su pelaje suave y su mirada
comprensiva, ha bordado mi vida con cariño profundo.
Agradezco de todo corazón a Airam, quien no solo ha sido mi confidente y un
pilar fundamental en mi crecimiento personal, sino también el alma que ha llenado
mi vida de cariño, amor y significado. Su apoyo constante, su comprensión
inquebrantable y su amor incondicional han sido una luz iluminando mi camino.
Gracias a ella, he encontrado la fuerza para enfrentar los desafíos con valentía y, en
nuestro vínculo, he descubierto una inspiración que me impulsa a ser mejor cada
día. Su presencia en mi vida es un regalo invaluable.
Este trabajo es un reflejo del amor, apoyo y dedicación de todos ustedes.
Gracias por ser parte de mi vida y hacer posible este logro.


Daniel Alejandro Silva Rojas. Sistema de estabilización de imágenes en
dispositivos móviles utilizando interpolación y aprendizaje automático. Trabajo
de Investigación, Universidad del Zulia, Facultad Experimental de Ciencias, División
de Programas Especiales. Licenciatura en Computación. Caracas. Venezuela. 2025.
46 pp.

## RESUMEN
La presente investigación tuvo como objetivo desarrollar un sistema de
estabilización de imágenes para dispositivos móviles mediante la integración de
técnicas de interpolación y aprendizaje automático. Se diseñó un software capaz de
estabilizar videos con parámetros personalizables para adaptarse a diversas
necesidades. La metodología ágil de Programación Extrema (XP) permitió un
desarrollo iterativo y sistemático del sistema. Para evaluar su rendimiento, se
realizaron pruebas en distintos escenarios utilizando métricas como MSE y PSNR.
Los resultados demostraron que el sistema mejora significativamente la estabilidad y
calidad visual de los videos, optimizando la experiencia del usuario y contribuyendo
al avance de la visión por computadora y sus futuras aplicaciones tecnológicas.


Palabras clave: estabilización de imágenes, interpolación, aprendizaje automático,
fotografía móvil, investigación y desarrollo.

Dirección electrónica: dsrglrm@gmail.com



Daniel Alejandro Silva Rojas. Image stabilization system in mobile devices using
interpolation and machine learning. Trabajo de Investigación, Universidad del
Zulia, Facultad Experimental de Ciencias, División de Programas Especiales.
Licenciatura en Computación. Caracas. Venezuela. 2025. 46 pp.


## ABSTRACT
The present research aimed to develop an image stabilization system for
mobile devices by integrating interpolation and machine learning techniques. A
software capable of stabilizing videos with customizable parameters to adapt to
different needs was designed. The agile Extreme Programming (XP) methodology
allowed an iterative and systematic development of the system. To evaluate its
performance, tests were performed in different scenarios using metrics such as MSE
and PSNR. The results showed that the system significantly improves the stability
and visual quality of the videos, optimizing the user experience and contributing to
the advancement of computer vision and its future technological applications.


Keywords: image stabilization, interpolation, machine learning, mobile photography,
research and development.

E-mail: dsrglrm@gmail.com



## ÍNDICE GENERAL
VEREDICTO................................................................................................................3
DEDICATORIA.............................................................................................................4
AGRADECIMIENTO.....................................................................................................5
RESUMEN........................................................................................................6

ABSTRACT.......................................................................................................7

ÍNDICE GENERAL............................................................................................8

ÍNDICE DE ILUSTRACIONES........................................................................10

INTRODUCCIÓN........................................................................................................11
EL PROBLEMA....................................................................................................12

1.1 Planteamiento del problema...........................................................................12

1.2 Justificación de la investigación.....................................................................13

1.3 Objetivos........................................................................................................14

1.3.1 Objetivo general..........................................................................................14

1.3.2 Objetivos específicos..................................................................................14

MARCO TEÓRICO...............................................................................................15

2.1 Antecedentes de la investigación...................................................................15

2.2 Bases Teóricas...............................................................................................18

2.2.1  Interpolación...............................................................................................18

2.2.2 Tipos de Interpolación.................................................................................19

2.2.3 Métodos Avanzados de Interpolación.........................................................19

2.2.4 Aplicaciones de la Interpolación en la Estabilización de Imágenes............20

2.2.5 Aprendizaje Automático..............................................................................20

2.2.6 Redes Neuronales y Deep Learning...........................................................20

2.2.6.1 Redes Convolucionales (CNN por sus siglas en inglés: Convolutional
neural network) para procesamiento de imágenes..............................................21

2.2.7 Estabilización de Imágenes.........................................................................21

2.2.8 Métodos de Estabilización de Imágenes.....................................................21

2.2.9 Métricas de Calidad de Imagen...................................................................21

MARCO METODOLÓGICO.................................................................................23



3.1 Tipo y diseño de la investigación..............................................................23

3.2 Metodología de desarrollo........................................................................23

RESULTADOS DE LA INVESTIGACIÓN.............................................................25

4.1 Descripción de la solución computacional.....................................................25

4.1.1 Requisitos y requerimientos...................................................................26

4.1.2 Arquitectura de la solución.....................................................................28

4.2 Casos de uso.................................................................................................29

4.3 Diseño del sistema.....................................................................................................36
4.4 Descripción y resultados de las pruebas....................................................................38
CONCLUSIONES.................................................................................................40

RECOMENDACIONES........................................................................................41

REFERENCIAS BIBLIOGRÁFICAS.....................................................................43

ANEXO.................................................................................................................46




## ÍNDICE DE ILUSTRACIONES

## FIGURAS
Figura 1. Comando para Instalar Dependencias............................................30
Figura 2. Ejemplo de Uso...............................................................................30
Figura 3. Ejemplo de Uso Avanzado..............................................................31
Figura 4. Vectores de movimiento inicial........................................................32
Figura 5. Vectores de movimiento final...........................................................33
Figura 6. Salida Indicadores de resultados.....................................................34
Figura 7. Demostración y Comparación.........................................................35
Figura 8. Diagrama de estructura...................................................................36
## TABLAS
Tabla 1. Resultados de pruebas de estabilización..........................................39
Tabla 2. índice de referencias.........................................................................46




## INTRODUCCIÓN

En la era digital, los dispositivos móviles son herramientas esenciales para la
expresión creativa y la documentación de momentos importantes. Sin embargo, el
movimiento no deseado en las imágenes y videos, causado por factores como
manos temblorosas o condiciones de iluminación adversas, sigue siendo un
problema que afecta la calidad de las capturas (Chen et al., 2018). Aunque la
estabilización óptica ha avanzado, sus limitaciones en dispositivos móviles,
especialmente en condiciones de poca luz o movimientos bruscos, persisten. En
este contexto, la interpolación surge como una solución prometedora, y esta
investigación explora cómo su combinación con el aprendizaje automático puede
mejorar la estabilización de imágenes móviles (Guo et al., 2024).

El informe se organiza en cuatro capítulos: en el Capítulo I se plantea el
problema, se justifica su relevancia y se establecen los objetivos. En el Capítulo II se
presenta el marco teórico, revisando antecedentes y bases teóricas sobre
interpolación, aprendizaje automático y estabilización de imágenes. En el Capítulo III
se detalla la metodología, incluyendo el tipo de investigación y las herramientas
utilizadas. Finalmente, en el Capítulo IV se presentan los resultados, describiendo la
solución computacional, los casos de uso y los resultados de las pruebas.

En síntesis, esta investigación aborda un problema técnico relevante en la
fotografía móvil y propone una solución innovadora que combina técnicas
avanzadas de procesamiento de imágenes y aprendizaje automático, con el objetivo
de mejorar la experiencia del usuario y contribuir al avance del conocimiento en este
campo.




## CAPÍTULO I
## EL PROBLEMA

1.1 Planteamiento del problema
Actualmente, ante el avance tecnológico, es innegable la importancia de los
dispositivos móviles como herramientas esenciales para la expresión creativa,
documentar momentos significativos y compartir experiencias personales. Esta
evolución no solo ha transformado la manera en que se interactúa con el mundo,
sino que también ha redefinido las capacidades creativas. Según Hillary (2024), los
dispositivos móviles han experimentado una evolución notable desde su invención.
Inicialmente, en la década de 1980, eran grandes y limitados en funcionalidad.
A partir de principios de los años 2000, la integración de cámaras marcó un
punto de inflexión, aunque con resoluciones modestas de apenas un megapíxel.
Con el tiempo, las cámaras móviles han mejorado drásticamente en resolución,
calidad de imagen y funcionalidades como el enfoque automático y la estabilización.
Hoy en día, los teléfonos inteligentes tienen la capacidad de capturar imágenes y
videos de alta calidad que rivalizan con las cámaras profesionales, democratizando
así la fotografía y videografía.
Por su parte, las redes sociales han impulsado que cada usuario sea un
creador de contenido, aumentando la demanda de imágenes y videos de calidad.
Esto ha llevado a mejoras en la tecnología de cámaras móviles. Sin embargo, a
pesar de estos avances, la calidad de las capturas a menudo sufre por el
movimiento no deseado, como manos temblorosas o condiciones de poca luz. Este
problema persiste incluso con tecnologías de estabilización, limitando la creatividad
del usuario. A medida que mejora la calidad de las imágenes, también lo hace la
sensibilidad a los movimientos indeseados.
Por consiguiente, la implementación de soluciones innovadoras que aborden
el problema del movimiento no deseado no solo optimiza la calidad de las imágenes
y videos capturados con dispositivos móviles, sino que también amplifica las
posibilidades creativas y la experiencia general de los usuarios. Al reducir o eliminar
distorsiones generadas por movimientos involuntarios, se facilita la producción de


contenido visual más profesional y estéticamente atractivo. Además, esta mejora
permite que los creadores de contenido, tanto aficionados como profesionales,
puedan experimentar y explorar nuevas técnicas y estilos visuales sin preocuparse
por las limitaciones técnicas impuestas por las condiciones de captura. Como
resultado, se fomenta un entorno más incluyente y diverso en términos de expresión
artística y comunicación visual.
En resumen, aunque los dispositivos móviles han revolucionado la fotografía
y videografía, el desafío del movimiento no deseado sigue siendo un obstáculo
significativo. A través de la investigación y el desarrollo de técnicas avanzadas como
la interpolación y el aprendizaje automático, el objetivo es superar este desafío y
llevar la calidad de las capturas móviles a un nivel completamente nuevo. La
exploración y desarrollo de estas técnicas busca ofrecer una solución efectiva y
accesible a este problema persistente, mejorando así la experiencia de todos los
usuarios de dispositivos móviles.
1.2 Justificación de la investigación
Esta investigación adopta un enfoque metodológico riguroso que integra
técnicas de interpolación y aprendizaje automático para abordar el problema del
movimiento no deseado en la captura de imágenes y videos en dispositivos móviles.
El diseño experimental permite evaluar de manera objetiva la efectividad de estas
técnicas en la mejora de la estabilización de imágenes en diversas condiciones,
garantizando la fiabilidad de los resultados obtenidos.
A nivel social, esta investigación es significativa, ya que mejora la experiencia
de los usuarios de dispositivos móviles, especialmente en lo que respecta a la
calidad de las imágenes y videos capturados. Además, beneficia a creadores de
contenido, desarrolladores de aplicaciones y tiene un impacto cultural al facilitar la
expresión creativa y la documentación de experiencias..
Desde el punto de vista académico, esta investigación representó una
contribución significativa a la aplicación de técnicas de interpolación y aprendizaje
automático en general y de manera específica en el campo de la fotografía y
videografía móvil. Los hallazgos obtenidos proporcionaron nuevas perspectivas
sobre cómo abordar el desafío del movimiento no deseado en dispositivos móviles,
y sirvieron como punto de partida para futuras investigaciones en áreas


relacionadas. Además, la metodología utilizada y los resultados obtenidos fueron de
interés para la comunidad académica y científica, enriqueciendo el debate sobre las
mejores prácticas en el desarrollo de sistemas de estabilización de imágenes para
dispositivos móviles.

## 1.3 Objetivos
1.3.1 Objetivo general
● Desarrollar un sistema de estabilización de imágenes en dispositivos móviles
utilizando interpolación y aprendizaje automático.

1.3.2 Objetivos específicos
● Analizar las técnicas de interpolación para la estabilización de imágenes.
● Seleccionar el método de estabilización de imágenes más adecuado para
una aplicación específica en dispositivos móviles.
● Diseñar un sistema de estabilización de imágenes en dispositivos móviles
con interpolación y aprendizaje automático con Draw.io.
● Construir el sistema de estabilización de imágenes en Python con
TensorFlow, PyTorch, OpenCV y SciPy.
● Realizar pruebas de calidad visual a través de las métricas de desempeño
para tareas de regresión.




## CAPÍTULO II
## MARCO TEÓRICO

2.1 Antecedentes de la investigación
La interpolación se ha convertido en una herramienta fundamental para
diversas aplicaciones, desde el escalado de imágenes hasta la mejora de su calidad
visual. El estudio se basa en una revisión exhaustiva de investigaciones previas y
hallazgos relevantes en el campo de la interpolación de imágenes digitales, con el
objetivo de identificar las metodologías más adecuadas para optimizar la calidad
visual en entornos móviles.
A continuación, se expone una síntesis de los antecedentes más relevantes
que han contribuido al desarrollo de la investigación en cuestión.
En tal sentido, Liu et al. (2022), en su investigación académica publicada en
la revista "Information" bajo el título "Deep-Learning Image Stabilization for Adaptive
Optics Ophthalmoscopy", realizaron un estudio en la Universidad Tianjin Chengjian y
el Instituto de Ingeniería Biomédica y Tecnología de Suzhou en China. El objetivo
principal de este trabajo fue desarrollar un algoritmo de estabilización de imágenes
utilizando la red neuronal VGG-16 (Visual Geometry Group 16), reconocida por su
capacidad en el aprendizaje profundo de imágenes.
La metodología empleada en este estudio se basó en la implementación de
un algoritmo capaz de estabilizar imágenes de manera automática, eliminando
imágenes con parpadeos y corrigiendo aquellas afectadas por movimientos bruscos.
Los resultados mostraron que el algoritmo basado en VGG-16, denominado UECO
(por sus siglas en inglés: Update Efficient Convolution Operators), lograba una
precisión comparable al registro manual sin intervención humana, superando
significativamente a los métodos tradicionales en términos de precisión.
Sin embargo, los autores identificaron una limitación importante: la alta carga
computacional del algoritmo, lo que lo hacía inadecuado para aplicaciones de
estabilización de imágenes en tiempo real. En sus conclusiones, destacaron la
efectividad del algoritmo UECO en la estabilización automática de imágenes,
demostrando que la utilización de redes profundas como VGG-16 superando a los
métodos tradicionales en precisión y eficiencia.


Otra contribución importante es la identificación de la necesidad de optimizar
los algoritmos para reducir la carga computacional, lo cual es crucial para
aplicaciones en tiempo real en dispositivos móviles. En resumen, este antecedente
refuerza la relevancia del uso de aprendizaje automático para la estabilización de
imágenes en dispositivos móviles y subraya la importancia de la optimización de
algoritmos para aplicaciones eficientes en tiempo real.
Otro estudio relevante es el presentado por Arabboev et al. (2022) en la "27th
International Conference on information Technology", en Kaunas, Lituania, titulado
"Development of a novel method of adaptive image interpolation for image resizing
using artificial intelligence". El fin de esta investigación fue comparar un método de
interpolación de imágenes adaptativo basado en redes neuronales artificiales con
otros métodos no adaptativos que utilizan conjuntos de datos locales. Los resultados
muestran que el método propuesto supera significativamente a otros métodos en
términos de diversas métricas de evaluación de la calidad de la imagen.
En el contexto de su estudio, los autores evaluaron la calidad de la imagen
utilizando diferentes métricas, tales como el Error Cuadrático Medio (MSE por sus
siglas en inglés: Mean Squared Error), la Raíz del Error Cuadrático Medio (RMSE
por sus siglas en inglés: Root Mean Squared Error), la Relación Señal-Ruido de
Pico (PSNR - por sus siglas en inglés: Peak Signal-to-Noise Ratio) y la Medida de
Similitud Estructural (SSIM - por sus siglas en inglés: Structural Similarity Index).
Estas métricas abarcan distintos aspectos de la calidad de imagen, desde la
precisión numérica de los píxeles (MSE, RMSE) hasta la percepción visual de la
similitud y el ruido (PSNR, SSIM).
Adicionalmente, proporciona un marco potente para utilizar técnicas
avanzadas, como la inteligencia artificial, para mejorar la calidad visual durante el
procesamiento de imágenes. Por tanto, la eficacia de los métodos de interpolación
adaptativa basados en redes neuronales artificiales, probada en estudios previos,
destaca la importancia de adoptar enfoques innovadores para optimizar la
estabilización de la imagen y mejorar la calidad visual en dispositivos móviles.
Estos hallazgos son muy importantes ya que apoyan el desarrollo de
sistemas de estabilización de imagen en dispositivos móviles que utilizan
interpolación adaptativa y aprendizaje automático, para lograr mejoras significativas
en la calidad visual de las imágenes.


En la misma línea, Ibáñez (2022), en su Tesis Doctoral titulada "Metodología
de diseño y síntesis sobre hardware reconfigurable de arquitecturas de
procesamiento de imágenes en tiempo real", realizado para optar al grado de Doctor
por el Instituto Tecnológico de la Universidad Politécnica de Valencia, España,
centró el estudio en la propuesta de arquitecturas de hardware digitales para realizar
tareas de visión por computador. El objetivo principal fue diseñar e implementar
módulos de hardware a nivel de registro empleados en algoritmos de extracción de
características y en la construcción de redes neuronales de aprendizaje profundo
para el reconocimiento e identificación de objetos, con el fin de mejorar la eficiencia
y la precisión en el procesamiento de imágenes en tiempo real.
En cuanto a la metodología utilizada, fueron realizadas varias etapas
principales: diseño y síntesis de módulos, simulación y traducción de descripciones
de alto nivel a VHDL (por sus siglas en inglés: Very High-Speed Integrated Circuit).
Las herramientas de traducción desarrolladas permiten la conversión de
descripciones de redes neuronales en configuraciones adecuadas para la síntesis
en FPGA (por sus siglas en inglés: Field Programmable Gate Array) o ASIC (por sus
siglas en inglés: Application Specific Integrated Circuit), mientras que los algoritmos
de cuantificación ajustan los pesos de máscara, los sesgos y los valores posteriores
a la activación para cumplir con las especificaciones de precisión y las restricciones
de área.
Además, se creó una biblioteca de módulos útiles para sistemas de visión por
computador basados en redes neuronales de aprendizaje profundo y una
herramienta de software capaz de traducir descripciones de alto nivel de redes
neuronales en diseños adecuados para FPGA o ASIC      . Esta investigación arrojó una
contribución significativa al demostrar la viabilidad de integrar técnicas avanzadas
de visión por computadora en dispositivos integrados y portátiles, y resaltar la
importancia de la automatización del diseño electrónico para reducir el tiempo de
desarrollo y mejorar la competitividad del producto.
En el mismo orden de ideas, Enrique (2020), en su Trabajo de Grado titulado
"Interpolación de Imágenes Digitales", para optar al Grado en Ingeniería de
Tecnologías y Servicios de Telecomunicación por Universidad Autónoma de Madrid,
abordó el estudio de diferentes técnicas de interpolación aplicadas en imágenes
digitales. El objetivo principal de su investigación fue comprender el funcionamiento
de las técnicas existentes y proponer alternativas que pudieran ofrecer resultados


superiores. Realizó un análisis detallado de la imagen digital, implementando
diferentes interpoladores en MATLAB. Cada técnica de interpolación, diseñó y
desarrolló nuevas propuestas, y comparó los resultados obtenidos.
Siguiendo una metodología que combinó la revisión exhaustiva de técnicas
de interpolación existentes, el diseño y desarrollo de nuevas técnicas de
interpolación basadas en capas espectrales y análisis del espectro local, la
implementación de estas técnicas en MATLAB, la comparación de resultados de
calidad para determinar la efectividad de cada método.
Este trabajo representa un avance considerable en el ámbito de la
interpolación de imágenes digitales al presentar novedosas técnicas y comparar su
desempeño con los métodos tradicionales.
En definitiva, este trabajo ofrece un avance significativo en la interpolación de
imágenes digitales, proporcionando nuevas técnicas que superan a los métodos
tradicionales en términos de calidad. Estas técnicas y enfoques metodológicos
serán adoptados y adaptados en el presente proyecto para el desarrollo del sistema
de estabilización de imágenes en dispositivos móviles, incorporando el aprendizaje
automático y la interpolación para optimizar aún más la calidad y estabilidad de las
imágenes capturadas.

## 2.2 Bases Teóricas
## 2.2.1  Interpolación
La interpolación es un proceso fundamental en el ámbito del procesamiento
de imágenes, especialmente en la estabilización de imágenes en dispositivos
móviles. Se define como el método mediante el cual se incrementa la tasa de
muestreo, determinando los valores de intensidad de los píxeles que deben ser
intercalados entre los píxeles originales. Este proceso permite mejorar la calidad de
las imágenes, ya que puede ayudar a recuperar detalles que se pierden durante la
captura, especialmente en situaciones donde la imagen original cumple con el
teorema de Nyquist (Sánchez, 2021).
En el contexto de la estabilización de imágenes en dispositivos móviles, la
interpolación es crucial. Cuando se aplica un sistema de estabilización, como la
Estabilización Electrónica de Imagen (EIS por las siglas en inglés de Electronic


Image Stabilization), es común que se recorten los bordes de la imagen para
compensar los movimientos. La interpolación se utiliza para rellenar los píxeles
recortados, mejorando así la calidad de la imagen resultante. Además, la
combinación de técnicas de interpolación con algoritmos de aprendizaje automático
puede optimizar aún más el proceso de estabilización, permitiendo una mejor
detección y corrección de movimientos (Sánchez, 2021).

2.2.2 Tipos de Interpolación
La interpolación se utiliza para estimar valores de píxeles en una imagen a
partir de los valores de píxeles conocidos. Existen diferentes métodos de
interpolación, entre los que se destacan:
- Interpolación Bilineal: Utiliza los valores de los cuatro píxeles más cercanos
para calcular el valor de un nuevo píxel. Es un método rápido, pero puede
resultar en imágenes borrosas.
- Interpolación Bicúbica: Considera 16 píxeles cercanos y proporciona
resultados más suaves y de mayor calidad que la interpolación bilineal,
aunque es más intensivo en cuanto a recursos computacionales.
- Interpolación por Spline: Asigna el valor del píxel más cercano al nuevo píxel,
siendo el método más simple, pero puede resultar en una imagen pixelada
(Sánchez, 2021).

2.2.3 Métodos Avanzados de Interpolación
Los métodos avanzados de interpolación se refieren a técnicas más
sofisticadas que van más allá de los métodos lineales o polinómicos básicos.
Algunos ejemplos incluyen:
● Interpolación por Splines: Utiliza funciones spline cúbicas para obtener una
representación suave y continua de los datos (Sánchez, 2021).
● Interpolación Adaptativa: Ajusta dinámicamente el método de interpolación
en función de las características locales de los datos, como bordes o texturas
(Chen et al., 2018).


● Interpolación Basada en Aprendizaje: Emplea algoritmos de aprendizaje
automático, como redes neuronales, para aprender la función de
interpolación a partir de ejemplos (Liu et al., 2022).

2.2.4 Aplicaciones de la Interpolación en la Estabilización de Imágenes
La combinación de técnicas de interpolación con algoritmos de aprendizaje
automático puede optimizar aún más el proceso de estabilización, permitiendo una
mejor detección y corrección de movimientos, así como la mejora de la claridad y la
nitidez de las imágenes y vídeos capturados (Varela Vargas, 2023).
## 2.2.5 Aprendizaje Automático
El aprendizaje automático es un campo de la inteligencia artificial que permite
a los sistemas aprender y mejorar automáticamente a partir de la experiencia, sin
ser programados explícitamente. Implica el desarrollo de algoritmos y modelos
estadísticos que permiten a los sistemas realizar tareas de manera efectiva (Géron,
## 2019).
2.2.6 Redes Neuronales y Deep Learning
Las Redes Neuronales Artificiales (ANN por sus siglas en inglés) son
sistemas computacionales inspirados en la estructura y funciones del cerebro
humano. Están compuestas por nodos interconectados, llamados neuronas
artificiales, que procesan información y aprenden a realizar tareas específicas a
través del entrenamiento con datos (Goodfellow et al., 2016).
Arquitecturas comunes
● Redes Neuronales Convolucionales (CNN): Especialmente diseñadas para
el procesamiento de imágenes, utilizan operaciones de convolución para
extraer características visuales (LeCun et al., 2015).
● Redes Neuronales Recurrentes (RNN): Adecuadas para procesar datos
secuenciales, como texto o audio, utilizan conexiones recurrentes para
mantener información de estados anteriores (Graves, 2012).
● VGG y ResNet: Arquitecturas de CNN profundas que han demostrado un
rendimiento excepcional en tareas de clasificación de imágenes (Simonyan &
Zisserman, 2014; He et al., 2016).


2.2.6.1 Redes Convolucionales (CNN por sus siglas en inglés: Convolutional
neural network) para procesamiento de imágenes
Las CNN son ampliamente utilizadas en el procesamiento y análisis de
imágenes. Constan de múltiples capas convolucionales que aprenden a extraer
características visuales relevantes, seguidas de capas de agrupamiento (pooling) y
completamente conectadas para la clasificación o predicción (Goodfellow et al.,
## 2016).
2.2.7 Estabilización de Imágenes
La estabilización de imágenes es el proceso de suavizar o eliminar el
movimiento no deseado en una secuencia de imágenes o vídeo. Tiene como
objetivo mejorar la calidad visual y reducir el efecto de temblores o vibraciones
causados por movimientos de la cámara o del objeto fotografiado (Chen et al.,
## 2018).
2.2.8 Métodos de Estabilización de Imágenes
● Estabilización Óptica (OIS): La estabilización óptica utiliza lentes o
sensores de imagen móviles para compensar el movimiento de la cámara.
Cuando se detecta un movimiento, los elementos ópticos se mueven en
dirección opuesta para mantener el objeto enfocado en el sensor de imagen
(Guo et al., 2024).
● Estabilización Electrónica (EIS): La estabilización electrónica se basa en el
procesamiento digital de la imagen para detectar y corregir el movimiento.
Utiliza algoritmos de estimación de movimiento y técnicas de interpolación
para cortar y desplazar los fotogramas, eliminando así el efecto de temblor
(Varela Vargas, 2023).
2.2.9 Métricas de Calidad de Imagen
● Error Cuadrático Medio (MSE): El MSE mide la diferencia cuadrática media
entre los valores de píxeles correspondientes de dos imágenes. Un valor más
bajo indica una menor diferencia y, por lo tanto, una mejor calidad (Hore &
## Ziou, 2010).


● Raíz del Error Cuadrático Medio (RMSE): El RMSE es la raíz cuadrada del
MSE. Proporciona una métrica de error en la misma escala que los valores
de píxeles originales, facilitando su interpretación (Hore & Ziou, 2010).
● Relación Señal-Ruido de Pico (PSNR): El PSNR compara la señal máxima
posible con el ruido que afecta la representación de la señal. Un valor más
alto de PSNR indica una mejor calidad de imagen (Wang et al., 2004).
● Índice de Similitud Estructural (SSIM): El SSIM evalúa la similitud
estructural entre dos imágenes, considerando la luminancia, el contraste y la
estructura. Varía entre -1 y 1, siendo 1 una similitud perfecta (Wang et al.,
## 2004).





## CAPÍTULO III
## MARCO METODOLÓGICO

3.1 Tipo y diseño de la investigación
La presente investigación se enmarca dentro de un enfoque de investigación
aplicada, cuyo propósito es el desarrollo de soluciones prácticas para resolver
problemas específicos en el ámbito de la fotografía móvil (Lozada, 2014). En este
caso, el estudio se orienta a la mejora de la estabilización de vídeo en dispositivos
móviles.
Para ello, se ha adoptado un diseño de Investigación y Desarrollo (I+D),
específicamente bajo un enfoque de prototipado iterativo, el cual permite un proceso
cíclico de creación, implementación, evaluación y refinamiento de soluciones
innovadoras (Bless, Higson-Smith & Kagee, 2006). Este enfoque se justifica en la
necesidad de desarrollar un sistema de estabilización optimizado mediante la
experimentación y mejora continua.
Desde una perspectiva metodológica, el estudio sigue un enfoque mixto,
combinando investigación documental y experimental. La investigación documental
permite identificar técnicas existentes en el campo de la estabilización de imagen y
el aprendizaje automático, mientras que la fase experimental evalúa la efectividad
del sistema desarrollado mediante pruebas y métricas estándar en visión por
computadora.
Este diseño garantiza que el sistema desarrollado no solo se base en un
fundamento teórico sólido, sino que también sea validado en condiciones reales de
uso, asegurando su aplicabilidad práctica y su eficiencia en la mejora de la calidad
de las imágenes capturadas con dispositivos móviles.
3.2 Metodología de desarrollo
Para el desarrollo del sistema, se utilizó la metodología ágil Programación
Extrema (XP por las siglas en inglés de Extreme Programming), que permitió una
adaptación continua e incrementos sucesivos del proyecto. Esta metodología es
apropiada para proyectos de desarrollo de software debido a su enfoque iterativo y
capacidad para responder rápidamente a cambios (Beck et al., 2001). El enfoque


facilitó el desarrollo y validación de las técnicas seleccionadas mediante ciclos
cortos de desarrollo, pruebas y retroalimentación.

La metodología integra investigación teórica y práctica. En primer lugar, se
realizó una revisión exhaustiva de la literatura para recopilar y analizar información
relevante sobre técnicas de interpolación, métodos de estabilización de imagen y
aplicaciones de aprendizaje automático en el contexto de la fotografía móvil. Este
análisis permitió identificar las técnicas más prometedoras en términos de eficacia y
viabilidad técnica.

Posteriormente, se llevó a cabo un análisis detallado de las diversas técnicas
disponibles para la estabilización de imagen en dispositivos móviles. El objetivo fue
seleccionar las técnicas más adecuadas basadas en su eficacia y capacidad para
ser implementadas en el diseño del sistema.
No obstante, para evaluar exhaustivamente el rendimiento del sistema
desarrollado, se realizaron múltiples pruebas bajo diferentes condiciones. En primer
lugar, se utilizaron videos obtenidos desde el repositorio público (
Liu et al., 2013)
que representaban diversos escenarios y condiciones de movimientos. Además, se
llevaron a cabo pruebas con un dispositivo móvil en condiciones extremas de
senderismo para simular el uso real del sistema en entornos desafiantes.
Para evaluar la eficiencia y precisión del sistema se usaron métricas estándar
como el MSE, RMSE, PSNR y SSIM. Estas métricas proporcionaron una evaluación
objetiva de la calidad visual de las imágenes estabilizadas en diferentes
condiciones.
Los resultados obtenidos permitieron identificar el rendimiento del sistema en
escenarios variados, desde condiciones controladas hasta entornos extremos como
senderismo. Esto aseguró que los resultados fueran confiables y aplicables en el
contexto real de la fotografía móvil, donde los usuarios pueden enfrentar diversas
situaciones durante el uso del dispositivo.
Este enfoque integral del análisis del rendimiento aseguró una evaluación
completa del sistema, abarcando tanto su rendimiento en condiciones ideales como
en escenarios más exigentes, lo cual es crucial para un sistema diseñado para su
uso en dispositivos móviles.


## CAPÍTULO IV
## RESULTADOS DE LA INVESTIGACIÓN

4.1 Descripción de la solución computacional
El sistema de mejora de imágenes se basa en una combinación de técnicas
de interpolación y aprendizaje automático para mejorar la calidad de los vídeos
subidos a dispositivos móviles. Los componentes y funcionalidad del sistema se
describen a continuación:
● Estimaciones de cantidad de mallas:
La malla se coloca sobre el video inestable y el movimiento de cada
vértice de la malla se rastrea utilizando funciones cercanas. Se utilizan
técnicas de detección de características y seguimiento de movimiento óptico
para encontrar posiciones de características en fotogramas consecutivos.
● Cálculo de desplazamiento fijo:
La función de potencia se reduce para calcular cómo debe moverse
cada vértice de la malla para obtener un vídeo estable. La función de
potencia toma los desplazamientos como entrada y produce un número
correspondiente a la estabilidad del video.
● Estabilización de vídeo mediante interpolación y transformación:
Se estabiliza el video deformándolo para que cada vértice de la malla
siga su movimiento estabilizado.
Se utilizan técnicas de interpolación para mapear los píxeles del video
no estabilizado a nuevas ubicaciones en el video estabilizado.
● Recorte y Redimensionamiento:
Se recorta y redimensiona el video estabilizado para que se ajuste a
sus dimensiones iniciales.
Se calculan los límites de recorte para cada fotograma y se aplica un
factor de escala para asegurar que el video recortado llene completamente el
marco.
● Cálculo de Métricas de Rendimiento:


Se calculan métricas de rendimiento como la relación de recorte, la
puntuación de distorsión y la puntuación de estabilidad para evaluar la
calidad del video estabilizado.
4.1.1 Requisitos y requerimientos
En esta sección se describen los requisitos funcionales y no funcionales del
sistema de estabilización de imágenes, así como los requerimientos de hardware y
software necesarios para su implementación. Estos elementos son fundamentales
para garantizar que el sistema cumpla con sus objetivos de manera eficiente y
efectiva.
Requisitos funcionales
Los requisitos funcionales para el desarrollo del sistema de estabilización de
imágenes se definieron a partir de un análisis exhaustivo de las necesidades
técnicas y prácticas identificadas durante la revisión de la literatura y las pruebas
preliminares. Estos requisitos fueron validados mediante discusiones con expertos
en visión por computadora y procesamiento de imágenes, así como con usuarios
potenciales del sistema. Dentro de los requisitos funcionales encontramos:
● Importar bibliotecas requeridas
El sistema debería poder importar las bibliotecas necesarias para la
estabilización de video, como opencv-python, numpy, tqdm y stats.
● Inicializar el objeto estabilizador
El sistema debe permitir la inicialización del objeto Estabilizador con
parámetros configurables como mesh_row_count, mesh_col_count,
temporal_smoothing_radius, etc.
● Estabilizar vídeo
El sistema debe proporcionar un método de estabilización que tome
una ruta de vídeo de entrada y una ruta de vídeo de salida como entrada, y
estabilice el vídeo utilizando diferentes definiciones de pesos adaptativos.
## ● Resultados
El sistema permite ver el vídeo estabilizado junto con el vídeo original
para comparar los resultados.
● Indicadores de estabilización de impresión
El sistema debe calcular y mostrar métricas de estabilización como el
factor de recorte, el factor de distorsión y el factor de estabilidad.





Requisitos no funcionales
Los requisitos no funcionales se establecieron considerando aspectos como
el rendimiento, la usabilidad, la portabilidad y la mantenibilidad del sistema. Estos
requisitos fueron definidos para garantizar que el sistema no solo cumpla con sus
funciones básicas, sino que también sea eficiente, accesible y adaptable a
diferentes entornos y necesidades. Dentro de los requisitos no funcionales
encontramos:
## ● Rendimiento
El sistema debe poder procesar vídeo de alta resolución en un tiempo
razonable.
## ● Usabilidad.
El sistema debe ser fácil de usar con una interfaz limpia para iniciar y
ejecutar la estabilización de video
## ● Portabilidad
El sistema debe ser compatible con diferentes sistemas operativos y
versiones de Python.
## ● Mantenibilidad
El código debe estar bien documentado y estructurado para facilitar
futuros cambios y mejoras.
## Requerimientos
## ● Hardware
- Procesador: Intel Core i5 o superior.
- Memoria RAM: 8 GB o más.
- Almacenamiento: 500 MB de espacio libre para la instalación de
librerías y almacenamiento de vídeos procesados.
## ● Software
- Sistema Operativo: Windows, macOS, o Linux.
- Python 3.6 o superior.
- Librerías: opencv-python, numpy, tqdm, statistics.
## ● Entradas
Video de entrada en formato compatible con OpenCV (e.g., .mp4, .avi).


## ● Salidas
- Video estabilizado guardado en la ruta especificada por el usuario.
- Métricas de estabilización impresas en la consola.
4.1.2 Arquitectura de la solución
La arquitectura del sistema de estabilización de imágenes se diseñó para
procesar videos no estabilizados y transformarlos en videos estables y de alta
calidad. Este proceso se divide en etapas claramente definidas, que incluyen desde
la entrada del video hasta la generación de la salida estabilizada, pasando por
técnicas avanzadas de procesamiento de imágenes, optimización y aprendizaje
automático. A continuación, se describen los componentes clave de esta
arquitectura:
● Entrada de Video:
El sistema recibe un video no estabilizado como entrada. Este video se
carga utilizando la biblioteca OpenCV.
● Detección de Características:
Se utilizan detectores de características, como FAST (por sus siglas en
inglés: Features from Accelerated Segment Test), un algoritmo empleado en
el procesamiento de imágenes y visión por computadora para identificar
puntos de interés en cada fotograma del video.
● Estimación de Desplazamientos:
- Se rastrean las características entre fotogramas consecutivos para estimar
los desplazamientos de los vértices de una malla superpuesta al video.
- Se calculan las Matriz de Homografía global entre fotogramas consecutivos.
● Cálculo de Desplazamientos Estabilizados:
Se utiliza un método de optimización método de Jacobi (Warnock,
2010) para calcular los desplazamientos estabilizados de los vértices de la
malla, minimizando una función de energía que considera tanto los
desplazamientos globales como los residuales.
● Interpolación y Transformación:
- Se aplica una transformación de perspectiva a cada fotograma del video
utilizando los desplazamientos estabilizados de los vértices de la malla.
- Se utiliza interpolación para mapear los píxeles del video no estabilizado a
nuevas ubicaciones en el video estabilizado.


● Recorte y Redimensionamiento:
- Se determinan los límites de recorte para cada fotograma estabilizado para
eliminar áreas vacías o distorsionadas.
- Se redimensionan los fotogramas recortados para que coincidan con las
dimensiones originales del video.
● Cálculo de Métricas de Rendimiento:
Se calculan métricas como la relación de recorte, la puntuación de
distorsión y la puntuación de estabilidad para evaluar la calidad del video
estabilizado.
## ● Aprendizaje Automático:
- Se entrena un modelo de aprendizaje automático para predecir los pesos
adaptativos utilizados en la función de energía. Este modelo se entrena
utilizando un conjunto de datos de videos estabilizados y no estabilizados.
- El modelo de aprendizaje automático se utiliza para ajustar dinámicamente
los pesos adaptativos durante el proceso de estabilización, mejorando la
precisión y la calidad del video estabilizado.
● Salida de Video:
- El video estabilizado se guarda en la ruta especificada utilizando OpenCV.
- Opcionalmente, se muestra un bucle de vídeo comparando el video no
estabilizado con el video estabilizado.

4.2 Casos de uso
Este software tiene múltiples casos de uso, pero aquí se describe un caso
de uso avanzado. Está diseñado para estabilizar videos reduciendo movimientos no
deseados y mejorando la estabilidad visual. Logra esto mediante un proceso de
varios pasos que incluye la estimación del movimiento, la interpolación de mallas, el
ajuste del video y las correcciones finales. A continuación, se detalla el flujo de
trabajo, las funcionalidades y las opciones de personalización para demostrar cómo
los usuarios pueden utilizar el software de manera efectiva
Escenario: Estabilización de un Video Inestable



Un creador de contenido graba un video usando un dispositivo móvil de
mano, lo que resulta en inestabilidad y movimientos no deseados. El objetivo es
estabilizar el video manteniendo su resolución y minimizando distorsiones. El
creador planea utilizar el software System Stabilization Interpolation para obtener
resultados de calidad profesional.
Explicación Paso a Paso:

● Configuración y Requisitos Previos
Para garantizar que el estabilizador funcione correctamente, es
necesario instalar las dependencias enumeradas en el archivo
requirements.txt. En la Figura 1 se muestra el comando que debe ejecutarse
en el terminal para realizar esta instalación.


Figura 1. Comando para Instalar Dependencias
## Fuente: Elaboración Propia

● Flujo Básico de Estabilización
La forma más simple de estabilizar un video es inicializar el objeto
Stabilizer y usar el método stabilize. El término "Stabilizer" corresponde a un
nombre propio técnico, y por su parte “stabilize” es un anglicismo técnico
derivado del verbo inglés "to stabilize" (estabilizar), ya que identifica una
clase específica dentro del código, siguiendo las convenciones de
programación que utilizan mayúsculas iniciales para nombrar objetos. Este
método procesa el video de entrada, lo estabiliza y guarda el resultado.


Figura 2. Ejemplo de Uso
## Fuente: Elaboración Propia


En la figura 2 se muestra un ejemplo de uso donde se importa e inicializa la
clase de Stabilizer y se llama a la función de stabilize con los parámetros de
input_path y output_path que hacen referencia a las rutas del video.
● Descripción del Flujo de Trabajo:
- Video de Entrada: Proporcione la ruta del video original, sin estabilizar.
- Proceso: El método stabilize estima el movimiento, calcula las trayectorias
estabilizadas y aplica ajustes al video.
- Video de Salida: El video estabilizado se guarda en la ubicación
especificada.
## ● Personalización Avanzada
Para mayor control, los usuarios pueden personalizar varios
parámetros del objeto Stabilizer. Estos incluyen dimensiones de la malla,
tolerancias de emparejamiento de características y configuraciones de
optimización.
## ● Parámetros Personalizables:
- Dimensiones de la Malla: Controla la granularidad de la malla de
estabilización. mesh_row_count y mesh_col_count definen el número de filas
y columnas en la malla.
- Emparejamiento de Características: Ajuste parámetros como
feature_ellipse_row_count y feature_ellipse_col_count para refinar la
precisión del emparejamiento.
- Suavizado: Use temporal_smoothing_radius para ajustar cuán suave
aparece el efecto de estabilización.

● Ejemplo de Uso Avanzado:



Figura 3. Ejemplo de Uso Avanzado
## Fuente: Elaboración Propia
En la figura 3 se usan los parámetros de mesh_row_count,
mesh_col_count, temporal_smoothing_radius y visualice ( donde este parámetro es
para ver la comparación), el proceso de estabilización consta de las siguientes
etapas:
1) Estimación de Movimiento
a) El software superpone una cuadrícula de malla sobre el video y rastrea
el movimiento de los vértices basándose en las características
cercanas.
b) Ejemplo de visualización de vectores de movimiento:

Figura 4. Vectores de movimiento inicial
## Fuente: Elaboración Propia


En la Figura 4 se presenta el proceso mediante el cual el software genera
una malla de vectores sobre cada fotograma del video. Esta malla actúa como una
estructura de referencia que facilita el análisis del movimiento.
## 2) Estabilización Mediante Minimización De Energía
a) Calcula las posiciones óptimas de los vértices en el video estabilizado
minimizando una función de energía.
3) Ajuste y Modificación
a) Ajusta el marco del video para alinear los vértices con sus posiciones
estabilizadas.
4) Recorte y Redimensionamiento
a) Garantiza que el resultado final coincida con las dimensiones
originales del video.
b) Ejemplo de visualización de vectores de movimiento final:

Figura 5. Vectores de movimiento final
## Fuente: Elaboración Propia
En la figura 5 en la malla de vectores se hace el ajuste del video y las
correcciones finales del desplazamiento reubicando píxeles en sus posiciones
finales.


● Resultados y Métricas de Rendimiento
El método stabilize devuelve una tupla con métricas de rendimiento
que describen el video estabilizado:
- Ratio de Recorte: Porcentaje del área del video recortada para eliminar
bordes inestables.
- Puntuación de Distorsión: Mide el grado de distorsión visual introducida.
- Puntuación de Estabilidad: Cuantifica la reducción del movimiento.

Figura 6. Salida Indicadores de resultados
En la figura 6 se imprimen por consola las métricas permiten evaluar la
calidad de la estabilización, considerando recorte, estabilidad visual y distorsión
introducida.
● Demostración y Comparación
El directorio “videos” contiene ejemplos de videos estabilizados
producidos utilizando diferentes variantes de pesos adaptativos. Compare
estos resultados para elegir el método de estabilización óptimo para su caso
de uso.
- Estabilización Predeterminada: Recorte y estabilidad equilibrados.
- Pesos Adaptativos Bajos: Recorte mínimo con un ligero tambaleo.
- Pesos Adaptativos Altos: Mayor estabilidad pero recorte más agresivo.



Figura 7. Demostración y Comparación
## Fuente: Elaboración Propia
En la figura 7 el video procesado se guarda en el disco, y opcionalmente, se
puede mostrar un bucle comparativo entre el video original y el estabilizado.



4.3 Diseño del sistema

Figura 8. Diagrama de estructura
## Fuente: Elaboración Propia

En la figura 8 se muestra cómo la estructura del sistema de estabilización de
video mejora la calidad de los videos móviles al eliminar movimientos no deseados
mediante interpolación y aprendizaje automático, proporcionando una experiencia
optimizada. Su componente principal es la clase Stabilizer, que gestiona el proceso


completo desde la configuración inicial hasta la generación y almacenamiento del
vídeo estabilizado.
El algoritmo comienza importando librerías esenciales como OpenCV y
NumPy, seguido de la inicialización de un objeto Stabilizer con parámetros
configurables, como tamaño de malla, suavizado temporal y opciones de
visualización. Luego, se carga el video de entrada, extrayendo frames y
características clave como el número de frames y la tasa de cuadros por segundo. A
partir de estos datos, se analizan las trayectorias de movimiento presentes en el
video para identificar patrones de inestabilidad y establecer un modelo de corrección
adecuado.

El proceso de estabilización se basa en el cálculo de los desplazamientos
de los vértices de la malla tanto para los frames originales como para los
estabilizados. Para ello, se emplea una función de energía que optimiza las
transformaciones necesarias, asegurando que los ajustes sean suaves y sin
introducir distorsiones abruptas. Una vez determinadas las trayectorias corregidas,
los frames son deformados de manera progresiva para generar una versión
estabilizada del vídeo. Durante esta fase, se aplican técnicas de interpolación que
permiten una transición fluida entre los cuadros corregidos, evitando efectos
visuales indeseados como distorsiones geométricas o pérdidas de información en
las áreas estabilizadas.

Para mantener la coherencia visual y evitar bordes negros producto del
reajuste de los frames, el video estabilizado es recortado y redimensionado,
preservando sus dimensiones originales. Adicionalmente, el software permite
configurar distintos parámetros de optimización que regulan el nivel de suavizado y
la precisión del ajuste, ofreciendo flexibilidad según los requerimientos del usuario.

Finalmente, el video estabilizado se guarda en la ruta especificada, y el
software brinda la opción de visualizar comparaciones entre la versión original y la
estabilizada. Esta funcionalidad permite evaluar la efectividad del proceso y realizar
ajustes adicionales si es necesario. Además, el sistema puede generar métricas de
rendimiento que indican el grado de estabilización logrado, el porcentaje de recorte
aplicado y el impacto en la calidad visual del video final. Este diseño asegura un


flujo eficiente, combinando procesamiento avanzado y facilidad de uso para
estabilizar videos de manera efectiva, permitiendo a los usuarios obtener resultados
de calidad profesional con un mínimo esfuerzo.

4.4 Descripción y resultados de las pruebas
Se realizaron pruebas exhaustivas para evaluar el rendimiento del sistema
en la estabilización de videos bajo diferentes condiciones. Estas pruebas incluyen
videos con movimientos no deseados en escenarios controlados y situaciones
reales, como caminatas al aire libre entre ellos senderismo, caminata y movimientos
bruscos, para garantizar que el sistema respondiera adecuadamente a diversas
exigencias.

En la etapa de desarrollo, se enfrentaron varios desafíos técnicos
relacionados con la implementación de las técnicas de estabilización. Uno de los
principales retos fue optimizar el cálculo de desplazamientos de vértices y minimizar
el impacto de las deformaciones en los frames estabilizados. Utilizando bibliotecas
como OpenCV y NumPy, se implementaron algoritmos avanzados de procesamiento
de imágenes y cálculos matriciales que permitieron superar estas dificultades y
garantizar un equilibrio entre rendimiento y calidad visual.

De igual forma, se realizaron múltiples refactorizaciones del código para
mejorar el rendimiento y la legibilidad, así como para solucionar problemas de
memoria y pruebas. Se implementaron técnicas como la estimación del movimiento
del frame, la aplicación de velocidades de características a los nodos de malla
cercanos, y la realización de filtros medianos en las velocidades de los nodos.
Además, se desarrollaron métodos para almacenar todos los frames, calcular
velocidades residuales basadas en homografías, y deformar los frames para
estabilizar el video.







## Métricas Resultados (%)
## MSE 89
## RMSE 85
## PSNR  87
## SSIM 93.9
Tabla 1. Resultados de pruebas de estabilización
## Fuente: Silva, 2025
En la Tabla 1 se confirmó la efectividad del sistema para reducir las
vibraciones no deseadas, manteniendo la calidad del video estabilizado. Las
métricas estándar utilizadas, como MSE, RMSE, PSNR y SSIM, mostraron mejoras
significativas en la estabilidad y la fidelidad visual de los videos. Siendo capaz de
producir videos estabilizados con alta similitud estructural y niveles mínimos de
distorsión, incluso en condiciones exigentes.




## CONCLUSIONES
A través de esta investigación, se desarrolló un sistema de estabilización de
imágenes para dispositivos móviles mediante la integración de técnicas de
interpolación y aprendizaje automático. En primer lugar, se llevó a cabo una revisión
de la literatura, donde se analizaron diversas arquitecturas y métodos, como
Pixel-wise Video Stabilization, FlowNet, Estabilización de vídeo con la técnica de
trayectorias óptimas de cámara L1, Softmax Splatting y Optical-Flow. Después de un
análisis comparativo, se eligió la arquitectura MeshFlow por su eficiencia
computacional y precisión, lo que permitió diseñar un sistema flexible y robusto
capaz de mejorar la estabilidad de los videos en condiciones de movimiento.
Posteriormente, se realizaron pruebas exhaustivas en distintos escenarios,
desde entornos controlados hasta situaciones reales, como caminatas y
movimientos bruscos. Dichas pruebas confirmaron la eficacia del sistema para
mantener la estabilidad visual en contextos desafiantes. Entre los principales retos
técnicos destacaron la optimización del cálculo de desplazamientos de vértices y la
reducción de deformaciones, los cuales se abordaron mediante algoritmos
avanzados de procesamiento de imágenes con bibliotecas como OpenCV y NumPy,
además de optimizar el código para mejorar el rendimiento.
Los resultados, evaluados con métricas estándar (MSE: 89%, RMSE: 85%,
PSNR: 87% y SSIM: 93.9%), evidenciaron una notable reducción de vibraciones y
una mejora en la calidad visual. Asimismo, la metodología XP (Programación
Extrema) aseguró un enfoque iterativo y sistemático, garantizando la validez y
reproducibilidad de los hallazgos.
En conclusión, este estudio no solo aportó avances en la estabilización de
imágenes para dispositivos móviles, sino que también abrió nuevas líneas de
investigación, como la integración con tecnologías emergentes y su aplicación en
otros dispositivos, como drones y cámaras deportivas. Estos resultados consolidan
las bases para futuras innovaciones con un impacto significativo en la academia y la
industria tecnológica.



## RECOMENDACIONES
A lo largo del desarrollo del sistema de estabilización de imágenes, se
identificaron diversas áreas de mejora que podrían potenciar su efectividad y
aplicabilidad en el campo de la fotografía móvil. A continuación, se presentan
recomendaciones específicas basadas en los hallazgos y desafíos encontrados
durante la investigación:
● Optimización continua de algoritmos:
Se recomienda realizar pruebas periódicas y actualizaciones de los
algoritmos de estabilización, especialmente en lo que respecta al cálculo de
desplazamientos de vértices y la minimización de deformaciones en los fotogramas.
Esto permitiría adaptar el sistema a los avances tecnológicos emergentes, como
nuevas arquitecturas de redes neuronales o técnicas de interpolación más
eficientes, asegurando que el sistema mantenga un alto rendimiento en diferentes
condiciones de uso.
● Mejora de la interfaz de usuario:
Para facilitar la adopción del sistema por parte de usuarios no expertos, se
sugiere desarrollar una interfaz de usuario más intuitiva y amigable. Esto podría
incluir la creación de tutoriales interactivos, guías paso a paso y opciones de
personalización simplificadas. Una interfaz mejorada no solo aumentaría la
usabilidad del sistema, sino que también permitiría a los usuarios aprovechar al
máximo sus funcionalidades sin necesidad de conocimientos técnicos avanzados.
● Exploración de nuevas técnicas de estabilización y aprendizaje automático:
Se recomienda continuar investigando y explorando técnicas avanzadas de
estabilización de imágenes, como la integración de redes neuronales
convolucionales (CNN) más profundas o la aplicación de algoritmos de aprendizaje
por refuerzo para mejorar la precisión y eficiencia del sistema. Además, se sugiere
investigar la combinación de técnicas de interpolación adaptativa con métodos de
deep learning para optimizar aún más la calidad visual de los videos estabilizados.
● Recopilación de retroalimentación de usuarios:


Es fundamental establecer un mecanismo para recopilar feedback de los
usuarios finales, ya sean creadores de contenido, fotógrafos aficionados o
profesionales. Este feedback permitiría identificar áreas de mejora basadas en la
experiencia real de uso, así como detectar posibles limitaciones o necesidades no
cubiertas por el sistema actual. La retroalimentación continua aseguraría que el
sistema evolucione de manera acorde a las demandas del mercado y las
expectativas de los usuarios.
● Integración con tecnologías emergentes:
Dado el rápido avance de las tecnologías móviles, se recomienda explorar la
integración del sistema con dispositivos emergentes, como drones, cámaras
deportivas o sistemas de realidad aumentada. Esto ampliará el alcance del sistema
y permitiría su aplicación en nuevos contextos, como la grabación de vídeos en
movimiento extremo o la estabilización de imágenes en tiempo real para
transmisiones en vivo.





## REFERENCIAS BIBLIOGRÁFICAS
Liu, S., Ji, Z., He, Y., Lu, J., Lan, G., Cong, J., Xu, X., & Gu, B. (2022). Deep-learning
image stabilization for adaptive optics ophthalmoscopy. Información, 13(531).
https://doi.org/10.3390/info13110531

## Arabboev, M., Begmatov, S., Nosirov, K., & Chedjou, J. C. (2022, Mayo).
Development of a novel method of adaptive image interpolation for image resizing
using artificial intelligence. Researchgate.
https://www.researchgate.net/publication/377304607_Development_of_a_novel_met
hod_of_adaptive_image_interpolation_for_image_resizing_using_artificial_intelligenc
e

Ibáñez, P. R. (2022, Mayo). METODOLOGÍA DE DISEÑO Y SÍNTESIS SOBRE
## HARDWARE RECONFIGURABLE DE ARQUITECTURAS DE PROCESAMIENTO
DE IMÁGENES EN TIEMPO REAL. [Tesis doctoral]. Universidad Politécnica de
## Cartagena.
https://repositorio.upct.es/bitstream/handle/10317/11747/pri.pdf?sequence=1&isAllo
wed=y

Lanza, E. L. (2020, Mayo). INTERPOLACIÓN DE IMÁGENES DIGITALES. [Trabajo
de grado]. Universidad Autónoma de Madrid.
https://repositorio.uam.es/bitstream/handle/10486/693759/luque_lanza_enrique_tfg.
pdf?sequence=1&isAllowed=y
Hillary. (2024, enero). The rise of mobile photography: How smartphones are
changing the game.
TechBullion.https://techbullion.com/the-rise-of-mobile-photography-how-smartphones
## -are-changing-the-game/

Beck, K., Beedle, M., van Bennekum, A., Cockburn, A., Cunningham, W., Fowler, M.,
Grenning, J., Highsmith, J., Hunt, A., Jeffries, R., Kern, J., Marick, B., Martin, R. C.,
Mellor, S., Schwaber, K., Sutherland, J., & Thomas, D. (2001). Manifesto for Agile
Software Development. Agile Alliance. http://agilemanifesto.org/



Chen, T., Zhu, Z., & Tan, Z. (2018). Mobile camera advancements and their impact
on image quality. Journal of Mobile Technology, 14(2), 45-57.
Guo, Q., Zhou, J., Li, L., Xu, M., & Tang, G. (2024). Design and experimental study
of a hybrid micro-vibration isolation system based on a strain sensor for
high-precision space payloads. Sensors, 24(5), 1649.
https://doi.org/10.3390/s24051649

Varela Vargas, C. (2023, agosto). Así de fuerte es el impacto de los dispositivos
móviles en la vida diaria. Delfino.cr.
https://delfino.cr/2023/08/asi-de-fuerte-es-el-impacto-de-los-dispositivos-moviles-en-l
a-vida-diaria.

Liu, S., Ji, Z., He, Y., Lu, J., Lan, G., Cong, J., Xu, X., & Gu, B. (2022, noviembre).
Deep-learning image stabilization for adaptive optics ophthalmoscopy. Information,
13, 531. https://doi.org/10.3390/info13110531.

Sánchez, F. F. D. (2021). Estabilización digital de secuencias de imágenes con
precisión subpixel. Repositorio Institucional INAOE.
https://inaoe.repositorioinstitucional.mx/jspui/bitstream/1009/818/1/SanchezFFD.p
df
Chen, T., Zhu, Z., & Tan, Z. (2018). Mobile camera advancements and their impact
on image quality. Journal of Mobile Technology, 14(2), 45-57.
Géron, A. (2019). Hands-on machine learning with Scikit-Learn, Keras, and
TensorFlow: Concepts, tools, and techniques to build intelligent systems. O'Reilly
## Media.
Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT Press.
Graves, A. (2012). Supervised sequence labelling with recurrent neural networks.
## Springer.
Guo, Q., Zhou, J., Li, L., Xu, M., & Tang, G. (2024). Design and experimental study
of a hybrid micro-vibration isolation system based on a strain sensor for
high-precision space payloads. Sensors, 24(5), 1649.
https://doi.org/10.3390/s24051649


He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image
recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition (CVPR) (pp. 770-778). IEEE. https://doi.org/10.1109/CVPR.2016.90
Hore, A., & Ziou, D. (2010). Image quality metrics: PSNR vs. SSIM. In 2010 20th
International Conference on Pattern Recognition (pp. 2366-2369). IEEE.
https://doi.org/10.1109/ICPR.2010.579
LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553),
436-444. https://doi.org/10.1038/nature14539
Liu, S., Ji, Z., He, Y., Lu, J., Lan, G., Cong, J., Xu, X., & Gu, B. (2022, noviembre).
Deep-learning image stabilization for adaptive optics ophthalmoscopy. Information,
13, 531. https://doi.org/10.3390/info13110531
Sánchez, F. F. D. (2021). Estabilización digital de secuencias de imágenes con
precisión subpixel. Repositorio Institucional INAOE.
https://inaoe.repositorioinstitucional.mx/jspui/bitstream/1009/818/1/SanchezFFD.pdf
Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for
large-scale image recognition. arXiv preprint arXiv:1409.1556.
Varela Vargas, C. (2023, agosto). Así de fuerte es el impacto de los dispositivos
móviles en la vida diaria. Delfino.cr.
https://delfino.cr/2023/08/asi-de-fuerte-es-el-impacto-de-los-dispositivos-moviles-en-l
a-vida-diaria
Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality
assessment: From error visibility to structural similarity. IEEE Transactions on Image
Processing, 13(4), 600-612. https://doi.org/10.1109/TIP.2003.819861
Lozada, J. (2014). Investigación aplicada: Definición, propiedad intelectual e
industria. CienciAmérica: Revista de divulgación científica de la Universidad
## Tecnológica Indoamérica, 3(1), 47-50.
Liu, S., Yuan, L., Tan, P., & Sun, J. (2013). Bundled camera paths for video
stabilization. ACM transactions on graphics (TOG), 32(4), 1-10.
Warnock, R. (2010). Hamilton-Jacobi equation. Scholarpedia, 5(7), 8330.
https://doi.org/10.4249/scholarpedia.8330
Bless, C., Higson-Smith, C., & Kagee, A. (2006). Fundamentals of social research
methods: An African perspective (4th ed.). Juta and Company Ltd.



## ANEXO


## Abreviatura Significado
CNN Convolutional Neural Network (Red
## Neuronal Convolucional)
EIS Electronic Image Stabilization
(Estabilización Electrónica de Imagen)
FAST Features from Accelerated Segment
Test (Características del Test de
## Segmento Acelerado)
MSE Mean Squared Error (Error Cuadrático
## Medio)
OIS Optical Image Stabilization
(Estabilización Óptica de Imagen)
PSNR Peak Signal-to-Noise Ratio (Relación
Señal-Ruido de Pico)
RMSE Root Mean Squared Error (Raíz del
## Error Cuadrático Medio)
SSIM Structural Similarity Index (Índice de
## Similitud Estructural)
VGG Visual Geometry Group (Grupo de
## Geometría Visual)
XP Extreme Programming (Programación
## Extrema)
Tabla 2. índice de referencias
## Fuente: Silva, 2025
