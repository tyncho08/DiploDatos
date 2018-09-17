# Reporte Ejercicio 1

## Modelos utilizados
1. __Modelo 1:__
1 capa oculta de 16 neuronas, con función de activación __ReLU__ y regularización __L2__. Salida compuesta por una capa de 1 neurona con activación __ReLU__ y regularización __L2__.
2. __Modelo 2:__
1 capa oculta de 32 neuronas, con función de activación __ReLU__ y regularización __L2__. Salida compuesta por una capa de 1 neurona con activación __ReLU__ y regularización __L2__.
3. __Modelo 3:__ 
1 capa oculta de 16 neuronas, con función de activación __ReLU__ y regularización __L2__. Una capa de Dropout, una capa oculta de 8 neuronas con activación __ReLU__ y regularización __L2__. Luego otra capa de Dropout y finalmente la capa de salida compuesta por una capa de 1 neurona con activación __ReLU__ y regularización __L2__.
4. __Modelo 4:__
1 capa oculta de 64 neuronas, con función de activación __ReLU__. Salida compuesta por una capa de 1 neurona con activación __ReLU__.
5. __Modelo 5:__
1 capa oculta de 16 neuronas, con función de activación __ReLU__ y regularización __L2__. Una capa oculta de 16 neuronas con activación __ReLU__ y regularización __L2__. Finalmente la capa de salida compuesta por una capa de 1 neurona con activación __ReLU__ y regularización __L2__.

## Procesado del Dataset mediante una representación TfIDf:
Parámetros del TfIDf:
1. __Bynary:__ True
2. __Ngram_range:__ (1,2)
3. __Stop_words:__ English
4. __Max_df:__ 0.7
5. __Norm:__ L2
2. __Vocabulary:__ None

## Decisiones de los Modelos
1. __Modelo 1:__ Se busco generar un modelo simple, con poco procesamiento para poder tomarlo como referencia para los otros dos modelos (Baseline).
   * __Cantidad de capas:__ 1 de entrada, 1 oculta y 1 de salida.
   * __Tamaño de las capas:__ 416701, 16 y 1.
   * __Activación:__ ReLU
   * __Regularización:__ L2.
   * __Dropout:__ No.
   * __Función de costo:__ binary_crossentropy.
   * __Optimizador:__ Adam.
   * __Metrica:__ Accuracy.
2. __Modelo 2:__ Se mantuvo la cantidad de capas, sin embargo se aumentó la cantidad de neuronas (Complejidad Media).
   * __Cantidad de capas:__ 1 de entrada, 1 oculta y 1 de salida.
   * __Tamaño de las capas:__ 416701, 32 y 1.
   * __Activación:__ ReLU.
   * __Regularización:__ L2.
   * __Dropout:__ No.
   * __Función de costo:__ binary_crossentropy.
   * __Optimizador:__ Adam.
   * __Metrica:__ Accuracy.
3. __Modelo 3:__ El modelo más complejo, se decidió utilizar una mayor cantidad de capas ocultas, junto con capas de Dropout.
   * __Cantidad de capas:__ 1 de entrada, 4 ocultas y 1 de salida.
   * __Tamaño de las capas:__ 416701, 16, dropout=0.5, 8, dropout=0.5 y 1.
   * __Activación:__ ReLU.
   * __Regularización:__ L2.
   * __Dropout:__ Si, a la salida de la capa de 32 neuronas y a la salida de la capa de 16 neuronas.
   * __Función de costo:__ binary_crossentropy.
   * __Optimizador:__ Adam.
   * __Metrica:__ Accuracy.
4. __Modelo 4:__ El modelo con más neuronas y sin regularización.
   * __Cantidad de capas:__ 1 de entrada, 1 oculta y 1 de salida.
   * __Tamaño de las capas:__ 416701, 64 y 1.
   * __Activación:__ ReLU.
   * __Regularización:__ None.
   * __Dropout:__ No.
   * __Función de costo:__ binary_crossentropy.
   * __Optimizador:__ Adam.
   * __Metrica:__ Accuracy.
5. __Modelo 5:__ El modelo con 2 capas ocultas iguales y regularización.
   * __Cantidad de capas:__ 1 de entrada, 2 ocultas y 1 de salida.
   * __Tamaño de las capas:__ 416701, 16, 16 y 1.
   * __Activación:__ ReLU.
   * __Regularización:__ L2.
   * __Dropout:__ No.
   * __Función de costo:__ binary_crossentropy.
   * __Optimizador:__ Adam.
   * __Metrica:__ Accuracy.

## Proceso de Entrenamiento
1. __División de Train y Test:__ 75% para Entrenamiento y 25% para Test.
2. __Tamaño de Batch:__ 50.
3. __Número de Épocas:__ 50.
4. __Métricas de evaluación:__ Accuracy.

## Overfitting
Para evitar el __overfitting__ se utilizó la técnica de __EarlyStopping__ de Keras, la cual ditiene el proceso de entrenamiento cuando una cantidad monitoreada ha dejado de mejorar, lo que evita que el modelo se __memorice__ los datos de entrenamiento.

Descripción de los parámetros del __EarlyStopping__:
1. __Monitor:__ Cantidad a monitorear.
2. __Min_delta:__ Cambio mínimo en la cantidad supervisada para calificar como una mejora, es decir, un cambio absoluto de menos de min_delta, contará como ninguna mejora.
3. __Patience:__ Número de épocas sin mejoría después del cual se detendrá el entrenamiento.
4. __Verbose:__ Modo de verbosidad.
5. __Mode:__ Uno de {auto, min, max}. En el modo mínimo, el entrenamiento se detendrá cuando la cantidad monitoreada haya dejado de disminuir; en modo máximo se detendrá cuando la cantidad monitoreada haya dejado de aumentar; en el modo automático, la dirección se deduce automáticamente del nombre de la cantidad supervisada.
6. __Baseline:__ Valor de referencia para alcanzar la cantidad monitoreada. El entrenamiento se detendrá si el modelo no muestra mejoras sobre la línea de base.

Parámetros de __EarlyStopping__ usados:
1. __Monitor:__ val_loss
2. __Min_delta:__ 0
3. __Patience:__ 15
4. __Verbose:__ 1
5. __Mode:__ auto
6. __Baseline:__ None

Además, se realizaron dos gráficas: __Accuracy vs Epoch__ y __Loss vs. Epoch__, para poder observar de mejor manera la evolución del aprendizaje de los modelos propuestos.

## Archivos Generados
1. __Modelos:__ Figuras de los modelos y los modelos en sí (Carpeta __Modelos__).
2. __Precisión y Otros:__ Reporte de los parámetros principales usados y precisiones logradas (Carpeta __Precision__).
3. __Resultados:__ Predicciones realizadas y gráficas de __Accuray vs. Epoch__ y __Loss vs. Epoch__ (Carpeta __Resultados__)
