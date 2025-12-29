#import "src/meta.typ": meta
#import "@preview/numbly:0.1.0": numbly
#import "@preview/tablem:0.2.0": tablem, three-line-table
#import "@preview/ctheorems:1.1.3": thmrules
#import "@preview/theorion:0.3.3"
#import "@preview/cetz:0.4.2"
#import "@preview/cmarker:0.1.7"
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *

#let conf(
  title: none,
  subtitle: none,
  semister: none,
  course: none,
  author: none,
  numbering: numbly("{1:一}、", default: "1.1  "),
  enum-numbering: "1.a.i.",
  lang: "zh",
  doc,
) = {
  set text(font: ("Libertinus Serif", "Songti SC", "SimSun"), lang: lang)
  show emph: text.with(font: ("Libertinus Serif", "STKaiti"))
  show: thmrules
  show: codly-init.with()

  set heading(numbering: numbering)

  set page(
    margin: (x: 4em, y: 6em),
    header: context {
      set text(0.9em)
      let page_number = here().page()
      stack(
        spacing: 0.2em,
        (
          if (page_number == 1) { align(right, emph(semister + " " + course)) } else {
            align(right, [#emph(author) | #emph(course) | #emph(title)])
          }
        ),
        v(0.3em),
        line(length: 100%, stroke: 1pt),
        v(.15em),
        line(length: 100%, stroke: .5pt),
      )
    },
    numbering: "1",
    footer: context {
      let page_number = here().page()
      let total_pages = counter(page).final().first()
      align(center)[Page #page_number of #total_pages]
    },
  )

  show ref: it => {
    let eq = math.equation
    let el = it.element
    if el != none and el.func() == eq {
      numbering(el.numbering, ..counter(eq).at(el.location()))
    } else {
      it
    }
  }
  show ref: r => text(blue, r)

  set enum(numbering: enum-numbering)

  show table.cell.where(y: 0): strong
  set table(
    align: center + horizon,
    inset: 0.6em,
    stroke: 0.05em,
  )
  show raw.where(block: true): set text(size: 0.8em)

  align(center, text(1.7em)[*#title —— #subtitle*])
  align(right, emph(author))
  doc
}

#show: doc => conf(
  doc,
  author: meta.author,
  course: meta.course,
  semister: meta.semester,
  title: meta.title,
  subtitle: meta.subtitle,
)

#include "src/main.typ"