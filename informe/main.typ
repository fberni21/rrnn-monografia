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

#lorem(500)

#figure(
    placement: auto,
    image("img/first_diagram.svg", width: 43.7%),
    caption: [],
) <fig:first-diagram>

#figure(
    placement: auto,
    image("img/second_diagram.svg", width: 75.7%),
    caption: [],
) <fig:second-diagram>

#figure(
    placement: auto,
    image("img/third_diagram.svg", width: 100%),
    caption: [],
) <fig:third-diagram>

// #bibliography("refs.bib", style: "institute-of-electrical-and-electronics-engineers")

// vim: ts=4 sts=4 sw=4 lbr
