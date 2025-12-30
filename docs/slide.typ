#import "src/meta.typ": *
#import "@preview/mitex:0.2.6": *
#import "@preview/touying:0.6.1": *
#import "@preview/numbly:0.1.0": *
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.10": *
#set text(font: ("Libertinus Serif", "Songti SC", "SimSun"), lang: "zh")
#show emph: text.with(font: ("Libertinus Serif", "STKaiti"))

#show: doc => {
  import themes.dewdrop: *
  show: dewdrop-theme.with(
    aspect-ratio: "16-9",
    footer: self => grid(
      columns: (1fr, 1fr, 1fr),
      align: center + horizon,
      self.info.institution, self.info.title, self.info.date.display(),
    ),
    navigation: "mini-slides",
    config-info(
      title: meta.slide-title,
      subtitle: meta.subtitle,
      author: meta.author,
      date: meta.date,
      institution: meta.institution,
      logo: none,
    ),
  )
  show: text.with(size: 0.75em)
  show: codly-init.with()
  show ref: r => text(blue, r)
  show raw.where(block: true): text.with(size: 0.8em)
  set heading(numbering: numbly("{1:一}、", default: "1.1  "))

  [
    #show: text.with(size: 1.5em)
    #title-slide()
  ]
  doc
  focus-slide[
    Thanks!
  ]
}

#include "src/main.typ"