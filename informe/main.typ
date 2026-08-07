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

El entrenamiento de las redes se hace sobre el conjunto de datos Fashion-MNIST. Este _dataset_ busca ser una alternativa más compleja a MNIST, el cual fue introducido por LeCun _et al._ en 1998 @lecun1998gradient. Fashion-MNIST consta de 60000 imágenes de entrenamiento y 10000 de evaluación, de $28 times 28$ píxeles en formato blanco y negro, las cuales muestran artículos de ropa distribuidos equitativamente entre las siguientes categorías:
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

La arquitectura original se estudia utilizando las técnicas de interpretabilidad, según se especifica en la @sec:interpretabilidad. En función de las observaciones, se diseña una segunda red de tamaño (cantidad de parámetros) similar a _FirstNet_, denominada _SecondNet_, cuya arquitectura se muestra en la @fig:second-diagram. _SecondNet_ tiene 21572 parámetros, distribuidos en cuatro capas convolucionales y una única capa completamente conectada (90, 1312, 5220 y 11700; y 3250 parámetros, respectivamente). Los filtros convolucionales son de tamaño $3 times 3$, el _stride_ es de 1 píxel, y el _padding_ es de 1 píxel a cada lado para las primeras dos capas, y nulo para las últimas dos.

Similarmente, estudiando con técnicas de interpretabilidad a _SecondNet_, se diseña una tercera red, denominada _ThirdNet_, cuya arquitectura se muestra en la @fig:third-diagram. _ThirdNet_ tiene 22346 parámetros, distribuidos en seis capas convolucionales y una única capa completamente conectada (40, 333, 1312, 3625, 5650 y 8136; y 3250 parámetros, respectivamente. Los filtros convolucionales son de tamaños y parámetros idénticos a los de _SecondNet_, con _padding_ de 1 píxel a cada lado para las primeras cuatro capas, y nulo para las últimas dos.

Por último, se diseñó una red sin la restricción de tamaño aplicada a las anteriores, denominada _LargeNet_, cuya arquitectura se muestra en la @fig:large-diagram. _LargeNet_ tiene 79873 parámetros, distribuidos en seis capas convolucionales y tres completamente conectadas (160, 2320, 3625, 5650, 8136 y 11700; 41600, 6192 y 490 parámetros, respectivamente). Las capas convolucionales tienen los mismos tamaños de fitros, _strides_ y _paddings_ que la _ThirdNet_.

#figure(
    placement: auto,
    scope: "parent",
    image("img/first_diagram.svg", width: 41.2%),
    caption: [Arquitectura de _FirstNet_.],
) <fig:first-diagram>

#figure(
    placement: auto,
    scope: "parent",
    image("img/second_diagram.svg", width: 71.3%),
    caption: [Arquitectura de _SecondNet_.],
) <fig:second-diagram>

#figure(
    placement: auto,
    scope: "parent",
    image("img/third_diagram.svg", width: 94.2%),
    caption: [Arquitectura de _ThirdNet_.],
) <fig:third-diagram>

#figure(
    placement: auto,
    scope: "parent",
    image("img/large_diagram.svg", width: 100%),
    caption: [Arquitectura de _LargeNet_.],
) <fig:large-diagram>

== Análisis de las redes
<sec:interpretabilidad>

Las redes convolucionales diseñadas se examinan utilizando redes deconvolucionales, con las técnicas desarrolladas en @zeiler2014. Para cada una de las arquitecturas, se construyó su correspondiente red deconvolucional. La red deconvolucional toma como entrada una activación intermedia de la red (la salida de alguna de las capas convolucionales), y la retrotrae hacia el espacio de los píxeles. Cada uno de los bloques que componen la red original se reemplaza por una función que intenta invertir la operación.

Los bloques de convolución se reemplazan por convoluciones traspuestas. Las rectificaciones ReLU se mantienen, garantizando que los mapas tengan activaciones no negativas. El submuestro, al no ser inversible, se reemplaza por una capa de _unpooling_ que utiliza la posición de la cual se extrajo originalmente el máximo valor y lo coloca en el correspondiente píxel, estableciendo los demás a cero. Este último reemplazo requiere que cuando se evalúa el conjunto de datos "hacia adelante" (usando la red original) se guarden los índices de los valores que activan cada máximo.

Siguiendo las ideas presentadas en @zeiler2014, se intentó comprender a qué aspectos de las imágenes la red le da mayor importancia para realizar la clasificación. Para ello, para cada filtro de cada capa se toma la activación más fuerte entre todas las imágenes, y se la proyecta sobre el espacio de píxeles usando la deconvolución. Como la red se entrena discriminativamente, las activaciones más fuertes se corresponden con las partes de la imagen que más ayudan a clasificarla. La visualización de los filtros de la _FirstNet_ se muestran en la @fig:first-deconv.

#figure(
    placement: auto,
    scope: "parent",
    caption: [Visualización de las activaciones más grandes de los filtros de la _FirstNet_, proyectados en el espacio de los píxeles por su correspondiente red deconvolucional. A la izquierda, se muestran las imágenes del conjunto de evaluación que se corresponden con las activaciones proyectadas a la derecha.]
)[
    #block(fill: gray, inset: 4pt, radius: 2pt)[
        *Capa 1*
        #image("img/first_0_deconv.svg", width: 50%)
    ]
    #block(fill: gray, inset: 4pt, radius: 2pt)[
        *Capa 2*
        #image("img/first_3_deconv.svg", width: 67%)
    ]
] <fig:first-deconv>

#bibliography("refs.bib", style: "institute-of-electrical-and-electronics-engineers")

// vim: ts=4 sts=4 sw=4 lbr wrap
