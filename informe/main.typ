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
    region: "AR",
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

Las redes neuronales convolucionales (CNNs) son herramientas muy eficaces en tareas de clasificación de imágenes. Esta arquitectura fue introducida en 1989 por LeCun _et al._ @lecun1989, y su uso se ha extendido durante las décadas posteriores gracias a los avances en _hardware_, así como con la aparición de nuevos conjuntos de entrenamiento. Sin embargo, durante mucho tiempo la elección exacta de la arquitectura y los hiperparámetros estuvo mayormente influida por heurísticas e intuiciones, debido a que no se comprendía su funcionamiento interno.

Desde la década de 2010, hubo avances significativos en la interpretabilidad de las redes neuronales convolucionales. Una herramienta muy útil fue desarrollada a partir de la red deconvolucional que proyecta las activaciones de las capas intermedias de la red original en los píxeles del espacio de entrada. Esta técnica fue inicialmente pensada para realizar aprendizaje no supervisado @zeiler2011, pero luego se usó para la interpretación de redes neuronales, mostrando qué partes de una imagen de entrada son responsables de una dada activación @zeiler2014. Una red deconvolucional se construye a partir de la inversión de los bloques funcionales que componen la CNN.

// TODO: completar introducción
// TODO: agregar introducción sobre mapas de Kohonen/SOM

= Desarrollo

Se emplean redes convolucionales clásicas, similares a las presentadas en @lecun1989, entrenadas con el objetivo de categorizar las imágenes del conjunto de datos Fashion-MNIST @xiao2017. Las redes están compuestas por una sucesión de bloques funcionales, los cuales pueden ser convolucionales, rectificadores, submuestreadores, o capas completamente conectadas. Las capas convolucionales aprenden un conjunto de filtros, los cuales se convolucionan con sus entradas. Los rectificadores aplican la función ReLU (definida como $ReLU(x) = max(x, 0)$) a sus entradas. Los submuestradores son _max-pool_, que subdividen la entrada en regiones sin solapamiento y eligen de cada una el valor más grande para reducir la dimensión. Las capas completamente conectadas finalmente clasifican la imagen en función de los _features_ extraídos por las capas anteriores.

El entrenamiento de las redes se hace sobre el conjunto de datos Fashion-MNIST. Este _dataset_ busca ser una alternativa más compleja a MNIST, el cual fue introducido por LeCun _et al._ en 1998 @lecun1998gradient. Fashion-MNIST consta de 60000 imágenes de entrenamiento y 10000 de evaluación, de $28 times 28$ píxeles en formato blanco y negro, las cuales muestran artículos de ropa distribuidos equitativamente entre diez categorías:
0. _T-shirt/top_ (remera/top),
1. _Trouser_ (pantalón),
2. _Pullover_ (pulóver),
3. _Dress_ (vestido),
4. _Coat_ (campera),
5. _Sandal_ (sandalia),
6. _Shirt_ (camisa),
7. _Sneaker_ (zapatilla),
8. _Bag_ (bolso),
9. _Ankle boot_ (bota),

Se entrena las redes utilizando retropropagación de errores (_error backpropagation_). La función de pérdida utilizada es la entropía cruzada, dado que la tarea es discriminativa. En todos los casos, se realizan diez épocas, entrenando con _batches_ de 128 muestras. Se utiliza el optimizador Adam @kingma2014adam, y una tasa de aprendizaje fija elegida empíricamente para cada modelo.

== Arquitectura de las redes

Inicialmente, se elige una red convolucional de estructura sencilla denominada _FirstNet_, la cual se muestra en la @fig:first-diagram. La red tiene 25034 parámetros, distribuidos en dos capas convolucionales con filtros cuadrados, y dos capas completamente conectadas (250 y 3690; 20244 y 850 parámetros, respectivamente). Los filtros convolucionales son de tamaño $7 times 7$ píxeles, y la convolución se realiza con un paso (_stride_) de 2 píxeles, agregando relleno (_padding_) de 3 píxeles. A la salida de cada capa convolucional se aplica la función ReLU.

La arquitectura original se estudia utilizando las técnicas de interpretabilidad, según se especifica en la @sec:análisis. En función de las observaciones, se diseña una segunda red de tamaño (cantidad de parámetros) similar a _FirstNet_, denominada _SecondNet_, cuya arquitectura se muestra en la @fig:second-diagram. _SecondNet_ tiene 21572 parámetros, distribuidos en cuatro capas convolucionales y una única capa completamente conectada (90, 1312, 5220 y 11700; y 3250 parámetros, respectivamente). Los filtros convolucionales son de tamaño $3 times 3$, el _stride_ es de 1 píxel, y el _padding_ es de 1 píxel a cada lado para las primeras dos capas, y nulo para las últimas dos.

Similarmente, estudiando con técnicas de interpretabilidad a _SecondNet_, se diseña una tercera red, denominada _ThirdNet_, cuya arquitectura se muestra en la @fig:third-diagram. _ThirdNet_ tiene 22346 parámetros, distribuidos en seis capas convolucionales y una única capa completamente conectada (40, 333, 1312, 3625, 5650 y 8136; y 3250 parámetros, respectivamente. Los filtros convolucionales son de tamaños y parámetros idénticos a los de _SecondNet_, con _padding_ de 1 píxel a cada lado para las primeras cuatro capas, y nulo para las últimas dos.

Por último, se diseñó una red sin la restricción de tamaño aplicada a las anteriores, denominada _LargeNet_, cuya arquitectura se muestra en la @fig:large-diagram. _LargeNet_ tiene 79873 parámetros, distribuidos en seis capas convolucionales y tres completamente conectadas (160, 2320, 3625, 5650, 8136 y 11700; 41600, 6192 y 490 parámetros, respectivamente). Las capas convolucionales tienen los mismos tamaños de fitros, _strides_ y _paddings_ que la _ThirdNet_.

#figure(
    placement: auto,
    scope: "parent",
    image("img/first_diagram.svg", width: 41.2%),
    caption: [Arquitectura de la _FirstNet_.],
) <fig:first-diagram>

#figure(
    placement: auto,
    scope: "parent",
    image("img/second_diagram.svg", width: 71.3%),
    caption: [Arquitectura de la _SecondNet_.],
) <fig:second-diagram>

#figure(
    placement: auto,
    scope: "parent",
    image("img/third_diagram.svg", width: 94.2%),
    caption: [Arquitectura de la _ThirdNet_.],
) <fig:third-diagram>

#figure(
    placement: auto,
    scope: "parent",
    image("img/large_diagram.svg", width: 100%),
    caption: [Arquitectura de la _LargeNet_.],
) <fig:large-diagram>

== Análisis de las redes

<sec:análisis>

Con el objetivo de mejorar la arquitectura inicialmente planteada en la _FirstNet_, se utilizan diferentes técnicas de análisis de redes. El método principal es la interpretabilidad de redes mediante redes deconvolucionales (_deconvnets_), presentado por Zeiler _et al._ en 2014 @zeiler2014. Las _deconvnets_ permiten entender cómo ajustar el tamaño y la distribución de los filtros de la red para obtener mejores resultados. El análisis se complementa estudiando la matriz de confusión @miller1955analysis, para entender cómo la red distribuye sus aciertos y errores de clasificación, y si hay clases que se confunden mayoritariamente entre sí. Adicionalmente, se usan mapas autoorganizados (_self organizing maps_, o SOM), introducidos por Kohonen en 1982 @kohonen1982self, para visualizar cómo se distribuyen las muestras en el espacio latente de alta dimensionalidad de las activaciones de una capa.

=== Redes deconvolucionales

Las redes convolucionales diseñadas se analizan utilizando redes deconvolucionales, con las técnicas desarrolladas en @zeiler2014. Para cada una de las arquitecturas, se construyó su correspondiente red deconvolucional. La red deconvolucional toma como entrada una activación intermedia de la red (la salida de alguna de las capas convolucionales), y la retrotrae hacia el espacio de los píxeles. Cada uno de los bloques que componen la red original se reemplaza por una función que intenta invertir la operación.

Los bloques de convolución se reemplazan por convoluciones traspuestas. Las rectificaciones ReLU se mantienen, garantizando que los mapas tengan activaciones no negativas. El submuestro, al no ser inversible, se reemplaza por una capa de _unpooling_ que utiliza la posición de la cual se extrajo originalmente el máximo valor y lo coloca en el correspondiente píxel, estableciendo los demás a cero. Este último reemplazo requiere que cuando se evalúa el conjunto de datos "hacia adelante" (usando la red original) se guarden los índices de los valores que activan cada máximo.

Siguiendo las ideas presentadas en @zeiler2014, se intentó comprender a qué aspectos de las imágenes la red le da mayor importancia para realizar la clasificación. Para ello, para cada filtro de cada capa se toma la activación más fuerte entre todas las imágenes, y se la proyecta sobre el espacio de píxeles usando la deconvolución. Como la red se entrena discriminativamente, las activaciones más fuertes se corresponden con las partes de la imagen que más ayudan a clasificarla. La visualización de los filtros de la _FirstNet_ se muestran en la @fig:first-deconv. Notar que en la primera capa, los filtros tienen campos receptivos de $7 times 7$ (el tamaño del filtro) por lo que se activan ante porciones pequeñas de la imagen original. En la segunda capa, la acción de combinar dos filtros en cascada y el _max-pooling_, agranda el campo receptivo hasta cubrir prácticamente toda la imagen.

#figure(
    placement: auto,
    scope: "parent",
    caption: [Visualización de las activaciones más grandes de los filtros de la _FirstNet_, proyectados en el espacio de los píxeles por su correspondiente red deconvolucional. A la izquierda, se muestran las imágenes del conjunto de evaluación que se corresponden con las activaciones proyectadas a la derecha.]
)[
    #block(fill: gray, outset: 4pt, radius: 2pt)[
        *Capa 1*
        #image("img/first_0_deconv.svg", width: 33%)
    ]
    #block(fill: gray, inset: 4pt, radius: 2pt)[
        *Capa 2*
        #image("img/first_3_deconv.svg", width: 50%)
    ]
] <fig:first-deconv>

Los filtros obtenidos para la _FirstNet_ muestran algunos inconvenientes. El comportamiento que se espera es que las primeras capas se activen ante aspectos de bajo nivel, como líneas, gradientes y patrones simples, mientras que las últimas capas combinan estos aspectos sencillos en formas complejas que la red usa para reconocer los objetos. Sin embargo, la capa 1 muestra tendencia a activarse ante patrones relativamente complejos, con contenidos de alta frecuencia. Una hipótesis planteada en este trabajo, basada en los resultados de @zeiler2014, es que el tamaño de los filtros de la primera capa es demasiado grande.

Por otro lado, la capa 2 de la red parece ser incapaz de reconstruir los objetos originales. Las activaciones de sus filtros parecen ruido, por lo que se cree que las posteriores capas completamente conectadas tienen que hacer mayor "trabajo" para discriminar los objetos. El comportamiento errático de los filtros se atribuye en parte a los filtros inadecuados de la primera capa, y en parte al _stride_ de 2 utilizado, que reduce la dimensionalidad de los datos agresivamente.

=== Matrices de confusión

Típicamente, el desempeño de un clasificador se evalúa usando su tasa de error, es decir, la proporción de muestras erróneamente clasificadas sobre el total de muestras evaluadas. Sin embargo, cuando se trabaja con múltiples clases esto no es suficiente dado que se pierde la información sobre si hay clases con tasas de error mayores que otras, y cuáles son. Para remediar esto, se utilizan las matrices de confusión @miller1955analysis. Una matriz de confusión es una matriz cuadrada con tantas filas y columnas como clases se buscan clasificar. Las filas representan las clases verdaderas, mientras que las columnas son las clases predichas por el modelo. En cada entrada, se coloca la cantidad de muestras que son de la clase correpondiente a la fila y que fueron clasificadas como pertenecientes a la clase de la columna. Por ejemplo, si la segunda fila son pantalones y la sexta columna son sandalias, en la dicha entrada se colocará la cantidad de pantalones que el modelo creyó que eran sandalias. Naturalmente, un modelo perfecto tendrá únicamente valores no nulos en la diagonal, ya que allí la clase real y la predicha coinciden.

En este trabajo se utiliza una variación de las matrices de confusión, donde se normalizan las cantidades en función de la clase verdadera. Es decir, cada fila se divide por la cantidad de muestras de su clase, de manera tal que la fila sume 100 %. De esta forma, se obtiene una tasa de aciertos para cada categoría, así como una tasa de error para cada uno de los posibles errores que puede cometer la red.

La matriz de confusión de la _FirstNet_ se muestra en la @fig:first-confusion. Si bien algunas categorías tienen una clasificación casi perfecta (zapatillas, bolsos y pantalones), las camisas tienen una tasa de error considerable. La matriz de confusión permite también ver con qué clases se confunde principalmente el modelo: las camisas incorrectamente clasificadas se confunen con remeras, pulóveres y camperas. Lograr una mejor distinción entre estas prendas será crucial si se quiere mejorar el desempeño global del modelo.

#figure(
    placement: auto,
    scope: "column",
    caption: [Matriz de confusión normalizada sobre las clases reales de la _FirstNet_.],
    image("img/first_confusion.svg", width: 100%)
) <fig:first-confusion>

=== Mapas autoorganizados

Los mapas autoorganizados de Kohonen son redes de neuronas, típicamente organizadas en una grilla bidimensional, que muestran un comportamiento emergente de auto-organización, preservando la topología de los datos de entrenamiento @kohonen1982self. Las neuronas se organizan de manera tal que las activaciones de neuronas vecinas son similares para eventos que se encuentran cerca en el espacio de entrada. Estas redes son útiles para visualizar datos de alta dimensionalidad, puesto que permiten transformar el espacio de entrada en una grilla bidimensional sin perder la estructura topológica inherente a los datos de entrada.

Las visualizaciones utilizadas en este trabajo son variaciones de los ejemplos mostrados en la documentación de la biblioteca `minisom` para implemetar SOM en Python @vettigli2018. La primera, a la que denominaremos _mapa de clases_, muestra el número de clase al que se corresponde cada muestra, ubicado en la posición de la neurona ganadora del mapa (aquella que más se activa ante esa entrada). La entrada del mapa son las activaciones de una de las capas de la red convolucional, para la muestra en cuestión. La @fig:first-som muestra el mapa de clases construido con las activaciones de la última capa convolucional de la _FirstNet_. Observar cómo algunas de las clases se agrupan en forma clara (por ejemplo la clase 1, los pantalones), indicando que las activaciones de la _FirstNet_ son similares para todos lo pantalones. Por el contrario, la clase 6 de las camisas no está agrupada, mostrando que la red convolucional fue incapaz de extraer _features_ que se activen de manera similar en todas las camisas.

#figure(
    placement: auto,
    scope: "column",
    caption: [Mapa de clases de las activaciones de la última capa convolucional de la _FirstNet_. Los ejes se corresponden con los índices bidimensionales de la grilla de neuronas del SOM. Cada dígito representa la clase de una imagen, en la posición de la neurona ganadora del SOM ante las activaciones de la entrada.],
    image("img/first_som.svg", width: 100%)
) <fig:first-som>

Una falencia de la técnica anterior es que se muestra únicamente la clase de las imágenes, sin referencia de cuál es la imagen original. Por ejemplo, no hay manera de entender visualmente por qué se forman dos agrupamientos para la clase 8 (los bolsos). La segunda visualización, denominada _mapa de imágenes_, muestra la imagen original ubicada en la posición de la neurona ganadora del mapa. Esto trae la ventaja de agregar información visual fácil de analizar, a costa de una imagen más grande y que carece de información de clases. El mapa de imágenes basado en las activaciones de la última capa convolucional de la _FirstNet_ se muestra en la @fig:first-som-imgs. Este segundo mapa revela información nueva: los dos agrupamientos de los bolsos se correponden con la presencia o no de una soga. A pesar de pertenecer a la misma clase, la red aprendió a darles representaciones sustancialmente diferentes a los bolsos con soga de aquellos que no la tienen.

#figure(
    placement: auto,
    scope: "parent",
    caption: [Mapa de imágenes de las activaciones de la última capa convolucional de la _FirstNet_. Las imágenes se encuentran posicionadas sobre la ubicación de la neurona ganadora del SOM ante las activaciones de esa entrada.],
    image("img/first_som_examples.svg", width: 85%)
) <fig:first-som-imgs>

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

// TODO: mejorar el caption
#figure(
    placement: auto,
    scope: "parent",
    caption: [Visualización de las activaciones más grandes de los filtros de la _ThirdNet_, proyectados en el espacio de los píxeles por su correspondiente red deconvolucional. A la izquierda, se muestran las imágenes del conjunto de evaluación que se corresponden con las activaciones proyectadas a la derecha.]
)[
    #grid(columns: (1fr, 1fr), gutter: 1em, align: horizon,
    grid.cell(
        block(fill: gray, inset: 4pt, radius: 2pt, width: 75%)[
            *Capa 1*
            #image("img/third_0_deconv.svg")
        ]
    ),
    grid.cell(
        block(fill: gray, inset: 4pt, radius: 2pt)[
            *Capa 2*
            #image("img/third_2_deconv.svg", width: 75%)
        ]
    ),
    grid.cell(
        block(fill: gray, inset: 4pt, radius: 2pt)[
            *Capa 3*
            #image("img/third_5_deconv.svg")
        ]
    ),
    grid.cell(
        block(fill: gray, inset: 4pt, radius: 2pt)[
            *Capa 4*
            #image("img/third_7_deconv.svg")
        ]
    ),
    grid.cell(
        block(fill: gray, inset: 4pt, radius: 2pt)[
            *Capa 5*
            #image("img/third_10_deconv.svg")
        ]
    ),
    grid.cell(
        block(fill: gray, inset: 4pt, radius: 2pt)[
            *Capa 6*
            #image("img/third_12_deconv.svg")
        ]
    ))
] <fig:third-deconv>

// TODO: mejorar el caption
#figure(
    placement: auto,
    scope: "parent",
    caption: [Visualización de las activaciones más grandes de los filtros de la _LargeNet_, proyectados en el espacio de los píxeles por su correspondiente red deconvolucional. A la izquierda, se muestran las imágenes del conjunto de evaluación que se corresponden con las activaciones proyectadas a la derecha.]
)[
    #grid(columns: (1fr, 1fr), gutter: 1em, align: horizon,
    grid.cell(
        block(fill: gray, inset: 4pt, radius: 2pt)[
            *Capa 1*
            #image("img/large_0_deconv.svg")
        ]
    ),
    grid.cell(
        block(fill: gray, inset: 4pt, radius: 2pt)[
            *Capa 2*
            #image("img/large_2_deconv.svg")
        ]
    ),
    grid.cell(
        block(fill: gray, inset: 4pt, radius: 2pt)[
            *Capa 3*
            #image("img/large_5_deconv.svg")
        ]
    ),
    grid.cell(
        block(fill: gray, inset: 4pt, radius: 2pt)[
            *Capa 4*
            #image("img/large_7_deconv.svg")
        ]
    ),
    grid.cell(
        block(fill: gray, inset: 4pt, radius: 2pt)[
            *Capa 5*
            #image("img/large_10_deconv.svg")
        ]
    ),
    grid.cell(
        block(fill: gray, inset: 4pt, radius: 2pt)[
            *Capa 6*
            #image("img/large_12_deconv.svg")
        ]
    ))
] <fig:large-deconv>

#bibliography("refs.bib", style: "institute-of-electrical-and-electronics-engineers")

// vim: ts=4 sts=4 sw=4 lbr wrap
