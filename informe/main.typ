#import "@preview/ctheorems:1.1.3": *
#show: thmrules

#set page(
    paper: "a4",
    margin: (x: 2.5cm, y: 2cm),
    columns: 2,
    numbering: "1",
    header: context {
        if counter(page).get().first() > 1 [
            _Franco Berni_
            #h(1fr)
            TP3: Redes de Kohonen
        ]
    }
)
#set text(
    font: "New Computer Modern",
    size: 10pt,
    lang: "es",
)
#set par(
    justify: true,
    first-line-indent: 1.5em,
    spacing: 0.65em,
)
#set block(
    spacing: 1.3em,
)

#show heading: smallcaps
#show heading: set text(weight: "regular")
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)")
#set figure(numbering: "1")
#set enum(indent: 1em)

#let definition = thmbox("definition", "Definición", inset: (x: 1.2em), base_level: 0)

#show ref: it => {
    let eq = math.equation
    let el = it.element
    if el == none or el.func() != eq { return it }
    link(el.location(), numbering(
        el.numbering,
        ..counter(eq).at(el.location())
    ))
}

#set document(title: [Mejora de Arquitecturas Convolucionales\ mediante Técnicas de Interpretabilidad])

#show title: smallcaps
#show title: set text(size: 17pt)
#show figure.caption: set text(size: 8pt)

#place(top + center, float: true, scope: "parent")[
    #image("img/fiuba.png", width: 60%)

    #title()
    #v(0.5em)

    Franco Berni \
    #link("mailto:fberni@fi.uba.ar") \
    110007

    #v(2em)
    #align(center)[#smallcaps[Resumen]]
    #text(style: "italic")[]
]

#let ReLU = math.op("ReLU")

= Introducción

Las redes neuronales convolucionales (CNNs) son herramientas de aprendizaje muy eficaces en las tareas de clasificación de imágenes. Esta arquitectura fue introducida en 1989 por LeCun _et al._~@lecun1989, y su uso se ha extendido durante las décadas posteriores gracias a los avances en _hardware_, así como con la aparición de nuevos conjuntos de entrenamiento. Sin embargo, durante mucho tiempo la elección exacta de la arquitectura y sus hiperparámetros estuvo mayormente influida por heurísticas e intuiciones, debido a que no se comprendía su funcionamiento interno.

Desde la década de 2010, hubo avances significativos en la interpretabilidad de las redes neuronales convolucionales. Por ejemplo, un avance provino de la red deconvolucional, que proyecta las activaciones de las capas intermedias de la red original en los píxeles del espacio de entrada. Esta técnica fue inicialmente pensada para realizar aprendizaje no supervisado~@zeiler2011, pero luego se usó para la interpretación de redes neuronales, mostrando qué partes de una imagen de entrada son responsables de una dada activación~@zeiler2014. Una red deconvolucional se construye a partir de la inversión de los bloques funcionales que componen la CNN.

// TODO: completar introducción
// TODO: agregar introducción sobre mapas de Kohonen/SOM

= Desarrollo
<sec:desarrollo>

Se emplean redes convolucionales clásicas, similares a las presentadas en~@lecun1989, entrenadas con el objetivo de categorizar las imágenes del conjunto de datos Fashion-MNIST~@xiao2017. Las redes están compuestas por una sucesión de bloques funcionales, los cuales pueden ser convolucionales, rectificadores, submuestreadores, o capas completamente conectadas (_fully connected_, o FC). Las capas convolucionales aprenden un conjunto de filtros, los cuales se convolucionan con sus entradas. Los rectificadores aplican la función ReLU (definida como $ReLU(x) = max(x, 0)$) a sus entradas. Los submuestradores son _max-pool_, que subdividen la entrada en regiones no solapadas y eligen de cada una el valor más grande para reducir la dimensión. Las capas completamente conectadas finalmente clasifican la imagen en función de los _features_ extraídos por las capas anteriores.

El entrenamiento de las redes se hace sobre el conjunto de datos Fashion-MNIST. Este _dataset_ busca ser una alternativa más compleja que el clásico MNIST, introducido por LeCun _et al._ en 1998~@lecun1998gradient. Fashion-MNIST consta de 60,000 imágenes de entrenamiento y 10,000 de evaluación, de $28 times 28$ píxeles en formato blanco y negro, las cuales muestran artículos de ropa distribuidos equitativamente entre diez categorías:
0. _T-shirt/top_ (remera/top),
1. _Trouser_ (pantalón),
2. _Pullover_ (pulóver),
3. _Dress_ (vestido),
4. _Coat_ (campera),
5. _Sandal_ (sandalia),
6. _Shirt_ (camisa),
7. _Sneaker_ (zapatilla),
8. _Bag_ (bolso), y
9. _Ankle boot_ (bota).

Se entrena las redes utilizando retropropagación de errores (_error backpropagation_)~@rumelhart1986. La función de pérdida utilizada es la entropía cruzada, dado que la tarea es discriminativa. En todos los casos, se entrena durante diez épocas, en _batches_ de 128 muestras. Se utiliza el optimizador Adam~@kingma2014adam, y una tasa de aprendizaje fija elegida empíricamente para cada modelo.

== Arquitectura de las redes

Inicialmente, se elige una red convolucional de estructura sencilla denominada _FirstNet_, la cual se muestra en la @fig:first-diagram. La red tiene 25,034 parámetros, distribuidos en dos capas convolucionales con filtros cuadrados, y dos capas completamente conectadas (250 y 3,690; 20,244 y 850 parámetros, respectivamente). Los filtros convolucionales son de tamaño $7 times 7$ píxeles, y la convolución se realiza con un paso (_stride_) de 2 píxeles, agregando relleno (_padding_) de 3 píxeles. A la salida de cada capa convolucional y la primera capa completamente conectada se aplica la función ReLU. Una capa de _max-pool_ se ubica entre las dos convolucionales.

#figure(
    placement: auto,
    scope: "parent",
    image("img/first_diagram.svg", width: 41.2%),
    caption: [Arquitectura de la _FirstNet_.],
) <fig:first-diagram>

La arquitectura original se estudia utilizando las técnicas de interpretabilidad, según se especifica en la @sec:análisis. En función de las observaciones, se diseña una segunda red de tamaño (cantidad de parámetros) similar a _FirstNet_, denominada _SecondNet_, cuya arquitectura se muestra en la @fig:second-diagram. _SecondNet_ tiene 21,572 parámetros, distribuidos en cuatro capas convolucionales y una única capa completamente conectada (90, 1,312, 5,220 y 11,700; y 3,250 parámetros, respectivamente). Los filtros convolucionales son de tamaño $3 times 3$, el _stride_ es de 1 píxel, y el _padding_ es de 1 píxel a cada lado para las primeras dos capas, y nulo para las últimas dos. Hay dos capas de _max-pool_, intercaladas entre las convolucionales. Los filtros de esta red y las sucesivas se renormalizan si su valor cuadrático medio supera 0.1, como sugiere @zeiler2014.

#figure(
    placement: auto,
    scope: "parent",
    image("img/second_diagram.svg", width: 62.4%),
    caption: [Arquitectura de la _SecondNet_.],
) <fig:second-diagram>

Similarmente, estudiando con técnicas de interpretabilidad a _SecondNet_, se diseña una tercera red, denominada _ThirdNet_, cuya arquitectura se muestra en la @fig:third-diagram. _ThirdNet_ tiene 22,346 parámetros, distribuidos en seis capas convolucionales y una única capa completamente conectada (40, 333, 1,312, 3,625, 5,650 y 8,136; y 3,250 parámetros, respectivamente). Los filtros convolucionales son de tamaños y parámetros idénticos a los de _SecondNet_, con _padding_ de 1 píxel a cada lado para las primeras cuatro capas, y nulo para las últimas dos. Hay dos capas de _max-pool_, ubicadas tras la segunda y cuarta capas convolucionales.

#figure(
    placement: auto,
    scope: "parent",
    image("img/third_diagram.svg", width: 94.2%),
    caption: [Arquitectura de la _ThirdNet_.],
) <fig:third-diagram>

Por último, se diseñó una red sin la restricción de tamaño aplicada a las anteriores, denominada _LargeNet_, cuya arquitectura se muestra en la @fig:large-diagram. _LargeNet_ tiene 79,873 parámetros, distribuidos en seis capas convolucionales y tres completamente conectadas (160, 2,320, 3,625, 5,650, 8,136 y 11,700; 41,600, 6,192 y 490 parámetros, respectivamente). Las capas convolucionales tienen los mismos tamaños de fitros, _strides_ y _paddings_ que la _ThirdNet_. La cantidad y posición de las capas de _max-pool_ también es la misma.

#figure(
    placement: auto,
    scope: "parent",
    image("img/large_diagram.svg", width: 100%),
    caption: [Arquitectura de la _LargeNet_.],
) <fig:large-diagram>

== Análisis de las redes

<sec:análisis>

Con el objetivo de mejorar la arquitectura inicialmente planteada en la _FirstNet_, se utilizan diferentes técnicas de análisis de redes. El método principal es la interpretabilidad mediante redes deconvolucionales (_deconvnets_), presentado por Zeiler _et al._ en 2014~@zeiler2014. Las _deconvnets_ permiten entender cómo ajustar el tamaño y la distribución de los filtros de la red para obtener mejores resultados. El análisis se complementa estudiando la matriz de confusión~@miller1955analysis, para entender cómo la red distribuye sus aciertos y errores de clasificación, y si hay clases que se confunden mayoritariamente entre sí. Adicionalmente, se usan mapas autoorganizados (_self organizing maps_, o SOM), introducidos por Kohonen en 1982~@kohonen1982self, para visualizar cómo se distribuyen las muestras en el espacio latente de alta dimensionalidad de las activaciones de una capa.

=== Redes deconvolucionales

Las redes convolucionales diseñadas se analizan utilizando redes deconvolucionales, con las técnicas desarrolladas en~@zeiler2014. Para cada una de las arquitecturas, se construyó su correspondiente red deconvolucional. La red deconvolucional toma como entrada una activación intermedia de la red (la salida de alguna de las capas convolucionales), y la retrotrae hacia el espacio de los píxeles. Cada uno de los bloques que componen la red original se reemplaza por una función que intenta invertir la operación.

Los bloques de convolución se reemplazan por convoluciones traspuestas. Las rectificaciones ReLU se mantienen, garantizando que los mapas tengan activaciones no negativas. El submuestreo, al no ser inversible, se reemplaza por una capa de _unpooling_ que utiliza la posición de la cual se extrajo originalmente el máximo valor y lo coloca en el correspondiente píxel, estableciendo los demás a cero. Este último reemplazo requiere que cuando se evalúa el conjunto de datos hacia adelante (usando la red original) se guarden los índices de los valores que activan cada máximo.

Siguiendo las ideas presentadas en~@zeiler2014, se intentó comprender a qué aspectos de las imágenes la red le da mayor importancia para realizar la clasificación. Para ello, para cada filtro de cada capa se toma la activación más fuerte entre todas las imágenes, y se la proyecta sobre el espacio de píxeles usando la deconvolución. Como la red se entrena discriminativamente, las activaciones más fuertes se corresponden con las partes de la imagen que más ayudan a clasificarla. La visualización de los filtros de la _FirstNet_ se muestran en la @fig:first-deconv. Notar que en la primera capa, los filtros tienen campos receptivos de $7 times 7$ (el tamaño del filtro) por lo que se activan ante porciones pequeñas de la imagen original. En la segunda capa, la combinación de dos filtros en cascada y el _max-pooling_, agrandan el campo receptivo hasta cubrir prácticamente toda la imagen.

#figure(
    placement: auto,
    scope: "parent",
    caption: [Visualización de las activaciones más grandes de los filtros de la _FirstNet_, proyectados en el espacio de los píxeles por su correspondiente red deconvolucional. A la izquierda, se muestran las imágenes del conjunto de evaluación que se corresponden con las activaciones proyectadas a la derecha.]
)[
    #grid(columns: (1fr, 1fr), gutter: 1em, align: horizon,
    grid.cell(
        block(fill: gray, outset: 4pt, radius: 2pt)[
            *Capa 1*
            #image("img/first_0_deconv.svg", width: 100%)
        ]
    ),
    grid.cell(
        block(fill: gray, inset: 4pt, radius: 2pt)[
            *Capa 2*
            #image("img/first_3_deconv.svg", width: 100%)
        ]
    ))
] <fig:first-deconv>

Los filtros obtenidos para la _FirstNet_ muestran algunos inconvenientes. El comportamiento que se espera es que las primeras capas se activen ante aspectos de bajo nivel, como líneas, gradientes y patrones simples, mientras que las últimas capas combinan estos aspectos sencillos en formas complejas que la red usa para reconocer los objetos. Sin embargo, la capa 1 muestra tendencia a activarse ante patrones relativamente complejos, con contenidos de alta frecuencia. Una hipótesis planteada en este trabajo, basada en los resultados de~@zeiler2014, es que el tamaño de los filtros de la primera capa es demasiado grande.

Por otro lado, la capa 2 de la red parece ser incapaz de reconstruir los objetos originales. Como los filtros se activan ante porciones casi aleatorias de las imágenes, se cree que las posteriores capas completamente conectadas tienen que hacer mayor trabajo para discriminar los objetos. El comportamiento errático de los filtros se atribuye en parte a los filtros inadecuados de la primera capa, y en parte al _stride_ de 2 píxeles utilizado, que reduce la dimensionalidad de los datos agresivamente.

=== Matrices de confusión

Típicamente, el desempeño de un clasificador se evalúa usando su tasa de error, es decir, la proporción de muestras erróneamente clasificadas sobre el total de muestras evaluadas. Sin embargo, cuando se trabaja con múltiples clases esto no es suficiente dado que se pierde la información sobre si hay clases con tasas de error mayores que otras, y cuáles son. Para remediar esto, se utilizan las matrices de confusión~@miller1955analysis. Una matriz de confusión es una matriz cuadrada con tantas filas y columnas como clases se buscan clasificar. Las filas representan las clases verdaderas, mientras que las columnas son las clases predichas por el modelo. En cada entrada, se coloca la cantidad de muestras que son de la clase correpondiente a la fila y que fueron clasificadas como pertenecientes a la clase de la columna. Por ejemplo, si la segunda fila son pantalones y la sexta columna son sandalias, en dicha entrada se colocará la cantidad de pantalones que el modelo creyó que eran sandalias. Naturalmente, un modelo perfecto tendrá únicamente valores no nulos en la diagonal, ya que allí la clase real y la predicha coinciden.

En este trabajo se utiliza una variación de las matrices de confusión, donde se normalizan las cantidades en función de la clase verdadera. Es decir, cada fila se divide por la cantidad de muestras de su clase, de manera tal que la fila sume 100~%. De esta forma, se obtiene una tasa de aciertos para cada categoría, así como una tasa de error para cada uno de los posibles errores que puede cometer la red.

La matriz de confusión de la _FirstNet_ se muestra en la @fig:first-confusion. Si bien algunas categorías tienen una clasificación casi perfecta (zapatillas, bolsos y pantalones), las camisas tienen una tasa de error considerable. La matriz de confusión permite también ver con qué clases se confunde principalmente el modelo: las camisas incorrectamente clasificadas se confunden principalmente con remeras, pulóveres y camperas. Lograr una mejor distinción entre estas prendas será crucial si se quiere mejorar el desempeño global del modelo.

#figure(
    placement: auto,
    scope: "column",
    caption: [Matriz de confusión normalizada sobre las clases reales de la _FirstNet_.],
    image("img/first_confusion.svg", width: 100%)
) <fig:first-confusion>

=== Mapas autoorganizados

Los mapas autoorganizados de Kohonen son redes de neuronas, típicamente organizadas en una grilla bidimensional, que muestran un comportamiento emergente de auto-organización, preservando la topología de los datos de entrenamiento~@kohonen1982self. Las neuronas se organizan de manera tal que las activaciones de neuronas vecinas son similares para eventos que se encuentran cerca en el espacio de entrada. Estas redes son útiles para visualizar datos de alta dimensionalidad, puesto que permiten transformar el espacio de entrada en una grilla bidimensional sin perder la estructura topológica inherente a los datos de entrada.

Las visualizaciones utilizadas en este trabajo son variaciones de los ejemplos mostrados en la documentación de la biblioteca `minisom` para implemetar SOM en Python~@vettigli2018. La primera la denominaremos _mapa de clases_.

#definition[
    Un _mapa de clases_ es una visualización que toma como entrada las activaciones de una capa de una red convolucional para cada una de las muestras de un conjunto de datos, y ubica en la posición de la neurona ganadora del SOM (la que más se activa ante dicha entrada) el número de clase al que se corresponde la muestra. El resultado es una grilla de números.
]

La @fig:first-som muestra el mapa de clases construido con las activaciones de la última capa convolucional de la _FirstNet_. Observar cómo algunas de las clases se agrupan en forma clara (por ejemplo la clase 1, los pantalones), indicando que las activaciones de la _FirstNet_ son similares para todos lo pantalones. Por el contrario, la clase 6 de las camisas no está agrupada, mostrando que la red convolucional fue incapaz de extraer _features_ que se activen de manera similar en todas las camisas.

#figure(
    placement: auto,
    scope: "column",
    caption: [Mapa de clases de las activaciones de la última capa convolucional de la _FirstNet_. Los ejes se corresponden con los índices bidimensionales de la grilla de neuronas del SOM. Cada dígito representa la clase de una imagen, en la posición de la neurona ganadora del SOM ante las activaciones de la entrada.],
    image("img/first_som.svg", width: 100%)
) <fig:first-som>

Una falencia de la técnica anterior es que se muestra únicamente la clase de las imágenes, sin referencia de cuál es la imagen original. Por ejemplo, no hay manera de entender visualmente por qué se forman dos agrupamientos para la clase 8 (los bolsos). La segunda visualización, que soluciona elegantemente este inconveniente, la denominaremos _mapa de imágenes_.

#definition[
    Un _mapa de imágenes_ es una visualización que toma como entrada las activaciones de una capa de una red convolucional para cada una de las muestras de un conjunto de datos, y ubica en la posición de la neurona ganadora del SOM (la que más se activa ante dicha entrada) la imagen de entrada. El resultado es una grilla de imágenes.
]

Esta visualización trae la ventaja de agregar información visual fácil de analizar, a costa de una imagen más grande y que carece de información de clases. El mapa de imágenes basado en las activaciones de la última capa convolucional de la _FirstNet_ se muestra en la @fig:first-som-imgs. Este segundo mapa revela información nueva: los dos agrupamientos de los bolsos se corresponden con la presencia o no de una soga. A pesar de pertenecer a la misma clase, la red aprendió a darles representaciones sustancialmente diferentes a los bolsos con soga de aquellos que no la tienen.

#figure(
    placement: auto,
    scope: "parent",
    caption: [Mapa de imágenes de las activaciones de la última capa convolucional de la _FirstNet_. Las imágenes se encuentran posicionadas sobre la ubicación de la neurona ganadora del SOM ante las activaciones de esa entrada.],
    image("img/first_som_examples.svg", width: 75%)
) <fig:first-som-imgs>

= Resultados

La @fig:tasas muestra los desempeños obtenidos para las diferentes redes neuronales convolucionales entrenadas en este trabajo.

#figure(
    placement: auto,
    scope: "column",
    caption: [Tasas de error de evaluación de los modelos entrenados, junto a su desvío estándar basado en cinco entrenamientos. El símbolo `(*)` indica que la diferencia es muy significativa ($p < 0.001$) respecto del modelo inmediatamente a su izquierda; mientras que `n.s.` indica que no es significativa ($p > 0.01$).],
    image("img/error_rates.svg", width: 100%)
) <fig:tasas>

Como se explicó en la @sec:desarrollo, primero se entrena la _FirstNet_ con el conjunto de datos Fashion-MNIST. La red obtenida logra una tasa de error del 12.0~%. En función de las observaciones hechas sobre la @fig:first-deconv, se determina que el tamaño de los filtros ($7 times 7$) y el _stride_ (2 píxeles) son inadecuados para el conjunto de datos en cuestión. Sin aumentar la cantidad de parámetros, se modifican los filtros a un tamaño de $3 times 3$ y el _stride_ se reduce a un píxel. Para poder aumentar la cantidad de capas convolucionales y de filtros, se decide remover una de las capas completamente conectadas. La hipótesis es que si la nueva red, la _SecondNet_ de la @fig:second-diagram, produce mejores representaciones internas al reducir el tamaño de los filtros, la capa final no necesitará demasiados parámetros para obtener resultados similares. Por otro lado, la matriz de confusión de la _FirstNet_ y el mal agrupamiento en el mapa de clases sugiere que el mayor lugar para mejoras está en la clasificación de las camisas. Aumentar el número de filtros por capa y el número de capas debería darle a la red mayo capacidad para extraer _features_ útiles para distinguir las camisas de las otras prendas similares.

La _SecondNet_, diseñada según las mejoras antes notadas, logra una tasa de error del 11.1~%. reduciendo en 0.93 puntos porcentuales respecto de la _FirstNet_ ($p < 0.0001$). Este resultado es más interesante si se considera que la _SecondNet_ tiene 13.8~% menos parámetros. Con el objetivo de determinar si los filtros son más apropiados y considerar posibles mejoras, se analizó por medio de _deconvnets_ a este segundo modelo. Las activaciones de los filtros de la _SecondNet_ se muestran en la @fig:second-deconv. Ahora es posible ver que la primera capa aprendió a reconocer líneas orientadas, principalmente verticales y horizontales, así como una diagonal. En cuanto a las siguientes capas, los filtros combinan las detecciones de bajo nivel de las capas anteriores para activarse ante figuras más complejas. La capa 2 se activa por ejemplo ante dos bordes paralelos para detectar mangas, o ángulos para detectar tacos. La tercera capa detecta figuras aún más complejas, como la parte central de las sandalias. La última se activa ante figuras casi completas.

Esta última capa y la anterior tienen filtros "muertos", que no se activan frente a ninguna parte de las imágenes (10 y 2 filtros, respectivamente). Esto probablemente se debe a la regularización que surge de la renormalización de filtros aplicada durante el entrenamiento. La regularización evita el sobreentrenamiento (_overfitting_), reduciendo la complejidad del modelo en forma automática.

// TODO: mejorar el caption
#figure(
    placement: auto,
    scope: "parent",
    caption: [Visualización de las activaciones más grandes de los filtros de la _SecondNet_, proyectados en el espacio de los píxeles por su correspondiente red deconvolucional. A la izquierda, se muestran las imágenes del conjunto de evaluación que se corresponden con las activaciones proyectadas a la derecha.]
)[
    #grid(columns: (1fr, 1fr), gutter: 1em, align: horizon,
    grid.cell(
        block(fill: gray, inset: 4pt, radius: 2pt)[
            *Capa 1*
            #image("img/second_0_deconv.svg")
        ]
    ),
    grid.cell(
        block(fill: gray, inset: 4pt, radius: 2pt)[
            *Capa 2*
            #image("img/second_3_deconv.svg")
        ]
    ),
    grid.cell(
        block(fill: gray, inset: 4pt, radius: 2pt)[
            *Capa 3*
            #image("img/second_6_deconv.svg")
        ]
    ),
    grid.cell(
        block(fill: gray, inset: 4pt, radius: 2pt)[
            *Capa 4*
            #image("img/second_8_deconv.svg")
        ]
    ))
] <fig:second-deconv>

El análisis de la matriz de confusión de la _SecondNet_, mostrada en la @fig:second-confusion, revela que la mejora de desempeño está principalmente dada por una mejor detección de las camisas (un aumento de 19 puntos porcentuales). Este resultado fue a costa de un ligero incremento de la tasa de error de algunas de las otras categorías, principalmente aquellas que son confundibles con las camisas, pero en suma la red presenta un desempeño mejor y más balanceado. Esto también se observa en el mapa de categorías de la @fig:second-som, donde la categoría de las camisas (6) presenta un cierto agrupamiento, aunque aún leve.

#figure(
    placement: auto,
    scope: "column",
    caption: [Matriz de confusión normalizada sobre las clases reales de la _SecondNet_.],
    image("img/second_confusion.svg", width: 100%)
) <fig:second-confusion>

#figure(
    placement: auto,
    scope: "column",
    caption: [Mapa de clases de las activaciones de la última capa convolucional de la _SecondNet_.],
    image("img/second_som.svg", width: 100%)
) <fig:second-som>

Los cambios que derivan en la _ThirdNet_ se ven motivados por los filtros de la primera capa de la _SecondNet_, donde cuatro de ellos aprenden variaciones de líneas verticales. Dada esta repetición, se reduce la cantidad de filtros en la capa 1 a apenas cuatro. Por otro lado, se hace a la red más profunda, redistribuyendo la cantidad de filtros en las demás capas para mantener la cantidad de parámetros aproximadamente constante. Esta nueva red alcanza una tasa de error de 10.2~%, una reducción de 0.93 puntos porcentuales respecto de la _SecondNet_ ($p = 0.0002$). La mejora es aceptable, dado que solo se usaron 3.6~% más parámetros que en la red anterior. Comparado a la _FirstNet_, el desempeño es considerablemente mejor (una tasa de error casi 2 puntos porcentuales menor sobre el 12~% inicial), con 10.7~% menos parámetros.

La inspección visual de las técnicas de interpretabilidad muestra resultados similares a los de la _SecondNet_. No se encuentran nuevos puntos de mejora, por lo que esta red fue la última entrenada con estos métodos. Como comparación, se entrenó un modelo considerablemente más grande, la _LargeNet_ de casi 80,000 parámetros. Se buscó ver cuánto más puede disminuirse la tasa de error utilizando simple fuerza bruta. Esta red obtiene una tasa de error de 9.8~%, cuya diferencia no es estadísticamente significativa comparada con la _ThirdNet_ ($p = 0.1644$). Esto sugiere que las técnicas de interpretabilidad empleadas para diseñar las redes logran un desempeño satisfactorio y casi tan bueno como una red mucho más grande. La matriz de confusión revela que el desempeño mejora principalmente en las remeras/tops, y en los pulóveres.

La _LargeNet_ cuenta con el número más grande de filtros en su primera capa y deja ver un fenómeno interesante. Las activaciones más grandes de la primera capa se muestran en la @fig:large-deconv. La mayor disponibilidad de filtros permitió que la red aprenda a reconocer más formas que líneas verticales y horizontales, haciendo mejor uso de ellos que en las redes anteriores. Hay filtros que reconocen patrones en vestidos, líneas diagonales en sandalias, líneas paralelas en la soga de un bolso, etc. Esto se contrapone con la hipótesis que se formuló tras ver los filtros de la _SecondNet_: la red aprendía redundantemente a reconocer líneas verticales y horizontales no porque tuviera demasiados filtros en su primera capa, sino porque no tenía la suficiente cantidad. Este aspecto contraintuitivo remarca la dificultad de interpretar cómo un cambio de arquitectura en una red afectará su desempeño, y motiva a continuar el estudio de técnicas de interpretabilidad.

En~@xiao2017 se citan algunos resultados obtenidos con diferentes arquitecturas y técnicas de aprendizaje para la tarea de clasificación del conjunto Fashion-MNIST#footnote[Esos resultados _no_ fueron validados por terceros.]. El mejor resultado citado es una tasa de error de 3.3~%, utilizando una _wide residual network_ @zagoruyko2016wide. Por otro lado, el mejor desempeño para redes convolucionales de menos de 100,000 parámetros es una tasa de error del 7.5~%.

#figure(
    placement: auto,
    scope: "column",
    caption: [Visualización de las activaciones más grandes de los filtros de la primera capa de la _LargeNet_, proyectados en el espacio de los píxeles por su correspondiente red deconvolucional. A la izquierda, se muestran las imágenes del conjunto de evaluación que se corresponden con las activaciones proyectadas a la derecha.]
)[
    #block(fill: gray, inset: 4pt, radius: 2pt)[
        *Capa 1*
        #image("img/large_0_deconv.svg", width: 100%)
    ]
] <fig:large-deconv>

= Conclusiones

// TODO: considerar agregar los resultados de las deconvoluciones, las matrices de confusión, y los mapas de clases/imágenes para las redes faltantes, dentro de un anexo para no estorbar la lectura, y solo dejar un par de ejemplos útiles dentro del cuerpo.

#bibliography("refs.bib", style: "institute-of-electrical-and-electronics-engineers")

// vim: ts=4 sts=4 sw=4 lbr wrap
