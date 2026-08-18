#import "@preview/euler-math:0.1.0": *

#set text(hyphenate: false)

// Cover page and configuration
#show: euler-math.with(
  title: [psD larueN],
  subtitle: [Project in machine learning at MATF],
  author: "Lazar Jovanović\nVasilije Ivanović",
)

#set text(hyphenate: true)

// Language
#set text(lang: "en")

// Table of contents
#outline(title: "Contents", indent: auto)

// Chapter pagebreak
#show heading.where(level: 1): it => {
  pagebreak()
  it
}

// Content

= The pitch

= Data

= Model

= Evaluation

= Conclusion

