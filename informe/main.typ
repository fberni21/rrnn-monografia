#set page(
    paper: "a4",
    margin: (x: 2.5cm, y: 2cm),
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
)

#show heading: smallcaps
#show heading: set text(weight: "regular")
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)")
#set figure(numbering: "1")

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

#align(center)[
    #image("img/fiuba.png", width: 60%)

    #title()
    #v(0.5em)

    Franco Berni \
    #link("mailto:fberni@fi.uba.ar") \
    110007
]

#v(2em)
#align(center)[#smallcaps[Resumen]]
#text(style: "italic")[]

Las redes neuronales convolucionales (CNNs) son herramientas muy eficaces en tareas de clasificación de imágenes. Esta arquitectura fue introducida en 1989 por LeCun et al. @lecun1989, y su uso se ha extendido durante las décadas posteriores gracias a los avances en _hardware_, así como con la aparición de nuevos conjuntos de entrenamiento. Sin embargo, durante mucho tiempo la elección exacta de la arquitectura y los hiperparámetros estuvo mayormente influida por heurísticas e intuiciones, debido a que no se comprendía su funcionamiento interno.

Desde la década de 2010, hubo avances significativos en la interpretabilidad de las redes neuronales convolucionales. Una herramienta muy útil fue desarrollada a partir de la red deconvolucional planteada en por Zeiler _et al._ en @zeiler2011, que proyecta las activaciones de las capas intermedias de la red original en los píxeles del espacio de entrada. Esta técnica fue inicialmente pensada para realizar aprendizaje no supervisado, pero luego se usaron para la interpretación de redes neuronales, mostrando qué partes de una imagen de entrada son responsables de una dada activación @zeiler2014. Una red deconvolucional se construye a partir de la inversión de los bloques funcionales que componen la CNN.



#figure(
    placement: auto,
    image("img/first_diagram.svg", width: 41.2%),
    caption: [],
) <fig:first-diagram>

#figure(
    placement: auto,
    image("img/second_diagram.svg", width: 71.3%),
    caption: [],
) <fig:second-diagram>

#figure(
    placement: auto,
    image("img/third_diagram.svg", width: 94.2%),
    caption: [],
) <fig:third-diagram>

#figure(
    placement: auto,
    image("img/large_diagram.svg", width: 100%),
    caption: [],
) <fig:large-diagram>

#bibliography("refs.bib", style: "institute-of-electrical-and-electronics-engineers")

// vim: ts=4 sts=4 sw=4 lbr wrap
