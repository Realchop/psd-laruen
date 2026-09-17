I need you to write a slide deck in Typst for a university ML course project. You do not have access to the code, so everything you need is below. Do not invent numbers, citations, or results — use only what is here.

## Who this is for

Two students at the Faculty of Mathematics, University of Belgrade (MATF), presenting a course project in machine learning. Authors: Lazar Jovanović and Vasilije Ivanović. Project name: **psD larueN**.

The audience is a professor and fellow students who know basic ML but nothing about audio or guitar amps. They are not audio researchers.

## What the project is

A **neural guitar amplifier**. You record a guitar plugged straight into an audio interface — a "dry" signal, thin and clean. A real amplifier turns that into a "wet" signal: distorted, filtered, coloured by the speaker cabinet. The project trains a neural network to imitate that transformation, so the amp becomes a piece of software. This is the same idea commercial products like Neural DSP sell.

The network learns a waveform-to-waveform mapping: 44,100 numbers per second in, 44,100 numbers per second out. It must also run in **real time**, because a guitarist has to hear the result while playing.

## The actual research question

WaveNet-style dilated causal convolutions have been the standard architecture for amp modelling since ~2019. Transformers have replaced convolutional and recurrent baselines nearly everywhere else. So:

> **Can a block-causal transformer match a WaveNet at the same parameter count and the same receptive field?**

The students built the transformer (called **AmpFormer**) specifically to test this, with WaveNet as the baseline.

**The answer is no — the transformer loses on every task, and the way it fails points at a specific cause.** This negative result *is* the contribution. Do not soften it or present it as a tie. The interesting part is the diagnosis, not the verdict.

## Data

- Source: the **IDMT-SMT-GUITAR V2** dataset (public, Zenodo) — real recordings of guitars played directly into an interface.
- The "amps" are not real hardware. Targets are generated in software with the `pedalboard` library, which gives an exact, reproducible ground truth for every input sample.
- **Four target chains**, chosen so they isolate different difficulties:
  - `clean` — does nothing (identity). A sanity check.
  - `cabinet` — a single convolution with a speaker impulse response. **Perfectly linear, but with a long tail.** Getting it right means getting phase right sample by sample.
  - `preamp` — noise gate, high-pass, 35 dB of distortion, some EQ. **Strongly nonlinear, but only a few milliseconds of memory.**
  - `metal` — `preamp` followed by `cabinet`. Both difficulties at once.
  - The point of this split: `cabinet` and `preamp` are single-stage and directly comparable, so they separate "long memory" from "nonlinearity" as sources of difficulty.
- **Train on one guitar (Ibanez 2820), validate on a different one (Career SG).** A different instrument is a real generalisation test; a random split of one recording would not be.
- Validation set: 43.7 minutes, ~2,596 segments.
- Training items: ~1 second segments (44,032 samples) plus a ~0.5 second "lead-in" of extra dry audio at the front. The lead-in is history — without it the first samples of every segment would have to be predicted from an all-zero past, which nothing can get right and which only adds noise to the gradient.
- Silent segments are dropped below -40 dBFS. The loss is relative to the target, so a near-silent target divides by almost nothing. This drops 1.3% of segments and cuts the spread of per-segment loss from ~950x to ~10x. Measured, not guessed.

## The two models

**WaveNet (baseline).** Dilated causal convolutions. Emits an output sample as soon as it has one input sample, so latency is essentially zero (0.02 ms).

**AmpFormer (the experiment).** A transformer, with three decisions worth a slide each:

1. **Attention cannot run at the sample rate.** One second of audio at 44.1 kHz is 44,100 positions; full self-attention over that is ~1.9 × 10⁹ pairs, about 7.8 GB of attention scores per head per second in float32, before gradients. It does not fit — this is not a tuning problem. So samples are grouped into **chunks of 64** before attention runs, bringing one second down to 689 positions. The grouping is a strided convolution (stride == kernel size), which is chunking plus a learned projection in one step.
2. **Causal, with a bounded window.** Attention is masked so a position sees only the past (a guitarist cannot be given audio depending on notes not yet played), and also only a bounded distance back — a *sliding* mask, not just causal. This keeps context comparable to a convolution's receptive field, and lets the two models be matched exactly rather than approximately.
3. **The model starts as a wire.** The output projection is zero-initialised and the raw input is added back, so at step zero the model is an exact passthrough. It never has to learn to reproduce the guitar — only the difference the amp makes to it. Both models do this, so it is not a confound.

**Fair comparison.** Both share the loss, the optimiser and schedule (AdamW + linear warmup + cosine decay), the seed, and the data. A learning-rate schedule is a confound: comparing a transformer on warmup-cosine against a convolution on a flat rate measures the schedule as much as the model.

| model | params | context | latency |
| --- | ---: | ---: | ---: |
| WaveNet 46 ms | 205,505 | 46.4 ms | 0.02 ms |
| AmpFormer 48 ms | 207,297 | 47.9 ms | 1.45 ms |
| WaveNet 186 ms | 243,265 | 185.7 ms | 0.02 ms |
| AmpFormer 187 ms | 207,297 | 187.2 ms | 1.45 ms |

At the short context: within 1% on parameters, 3% on context. At the long context **WaveNet has 17% more parameters — the advantage goes to the baseline, not to AmpFormer.** AmpFormer's 1.45 ms latency is structural (it cannot emit a chunk until all 64 samples arrive) but is still below what a player notices.

## Results

Metric is **ESR** (error-to-signal ratio) on the *validation guitar*. Lower is better. Explain ESR in one line: squared error divided by the energy of the target, so it is scale-independent.

| task | context | WaveNet | AmpFormer | gap |
| --- | ---: | ---: | ---: | ---: |
| cabinet *(linear, long tail)* | 46 ms | 0.00010 | 0.00967 | 97x |
| cabinet | 186 ms | 0.00004 | 0.01042 | 260x |
| preamp *(nonlinear, short)* | 46 ms | 0.00091 | 0.00695 | 8x |
| preamp | 186 ms | 0.00022 | 0.00770 | 35x |
| metal *(preamp + cabinet)* | 46 ms | 0.00622 | 0.01263 | 2x |
| metal | 186 ms | 0.00075 | 0.00565 | 8x |

`clean` is left out: both models solve it trivially given the residual connection, so it separates nothing.

**AmpFormer loses on every task at every context length.**

There is also a second metric, **aliasing-to-signal ratio** — distortion creates harmonics above the Nyquist frequency, which fold back as inharmonic components and sound like harshness or a metallic ring rather than ordinary error. Two models at the same ESR can differ audibly in exactly this way. The gap here is much larger than the ESR gap:

| task | context | WaveNet | AmpFormer |
| --- | ---: | ---: | ---: |
| cabinet | 186/187 ms | 0.00000006 | 1.51 |
| preamp | 186/187 ms | 0.020 | 0.078 |
| metal | 186/187 ms | 0.00049 | 0.167 |

## The diagnosis — this is the heart of the talk

**Observation 1: long memory costs more than nonlinearity.** Against WaveNet, AmpFormer is roughly an order of magnitude further behind on `cabinet` (97–260x) than on `preamp` (8–35x). The linear task with a long tail hurts it far more than the strongly nonlinear one.

**Observation 2: more context does not help it.** Going from ~46 ms to ~186 ms of context:

| task | WaveNet | AmpFormer |
| --- | ---: | ---: |
| cabinet | 2.5x better | **0.9x — worse** |
| preamp | 4.1x better | **0.9x — worse** |
| metal | 8.3x better | 2.2x better |

A model starved of context benefits from more of it. This one does not — on two of three tasks it gets *worse*. Whatever limits AmpFormer, it is not reach.

**Conclusion: the chunk is the bottleneck.** The model compresses 64 samples into one 64-dimensional vector and reconstructs 64 samples from it. For a nonlinearity that is survivable — the residual connection already carries the waveform the distortion curve acts on. For a linear filter with a long tail it is not: **sample-accurate phase cannot be recovered from a per-chunk summary, and no amount of extra context restores resolution the bottleneck has already removed.** The information lost is *within* a chunk, and context length does not address it.

**The fix this points to:** stop predicting samples from chunk summaries. Predict **filter coefficients** per chunk instead, and apply the filtering at the sample rate. Attention stays cheap (still 689 positions per second) while the output regains sample resolution. This is untested — present it as the next thing to try, not as a result.

One caveat worth one honest line: `metal` is an apparent counter-example — it contains the same cabinet, yet shows the *smallest* gap. The likely reason is that ESR normalises by target energy and 35 dB of distortion dominates that energy, so the same absolute filtering error reads as a smaller relative one. **This has not been verified.** Observation 2 does not depend on it.

## Honesty slide (keep it, keep it short)

- The reported figures are the *best epoch* selected on validation, so they are an optimistic upper bound on quality. The gaps (2x to 260x) are far too large to be explained by this.
- Validation doubles as model selection — no separate test split.
- One seed per configuration.
- No chunk-size ablation. Training at chunk_size 16 and 32 would test the bottleneck claim directly. This is the single most valuable missing experiment.
- **Not a gap:** AmpFormer was not undertrained. Both models ran under identical early stopping (patience 40 epochs), and on the two tasks that matter AmpFormer actually trained *longer* in three of four cells, by as much as 4x.

## What I want you to produce

A **Typst** presentation, delivered as a single `.typ` file I can compile with `typst compile`.

Hard requirements:

1. **No external Typst packages.** Do not `#import "@preview/..."` anything. Use only built-in Typst. Set up 16:9 slides with `#set page(paper: "presentation-16-9", ...)` and separate slides with `#pagebreak()`. I need this to compile with no network access and no package downloads.
2. Write a small number of your own helper functions at the top (e.g. one `slide(title, body)` function) so the body of the file reads as a clean list of slides. Keep the styling code short — this is a student talk, not a design exercise.
3. **Language: English.**
4. **Length: about 12 slides.** The slot is 15 minutes, but roughly 4 of those go to a live demo (a friend plays guitar through the model in real time). So the slides need to fit ~10-11 minutes of speaking.
5. Include a **dedicated demo slide** — a near-empty slide that acts as a cue to stop talking and switch to the live playing.
6. Every slide gets **speaker notes**: a short paragraph, placed in the `.typ` as a comment block (`//`) directly above that slide, saying what to actually say out loud. The notes must not render in the PDF.
7. Bullets on slides should be **short** — a few words, not sentences. The sentences go in the speaker notes. A slide with a table is allowed to be denser.
8. Use the tables above as real Typst tables. The three that matter most: the matched-models table, the ESR results table, and the context-scaling table. Bold or otherwise mark the two "worse" cells in the context-scaling table — that is the punchline of the whole talk.

Tone and level:

- **Plain and direct.** This is a student course project, not a paper submission. No grandiose framing, no "we propose a novel...", no fake significance.
- **Do not over-explain basics.** Assume the audience knows what a neural network, a convolution, attention, and a train/validation split are. Do not spend slides defining them. Do explain the *audio* side briefly, since that is the part they will not know: dry vs wet, why sample rate makes attention expensive, what a speaker cabinet does, what ESR is.
- Lead with the question and the answer. The audience should know by slide 3 that the transformer lost; the rest of the talk is why.

Suggested shape, adjust if you have a better one:

1. Title
2. What a guitar amp does / dry vs wet — the problem in one picture's worth of words
3. The question, and the answer up front (transformer loses)
4. Data: four targets, and why those four
5. Train on one guitar, validate on another
6. WaveNet baseline, in brief
7. AmpFormer: why chunking is forced, and what it costs
8. How the comparison was made fair (matched table)
9. Results (ESR table)
10. Diagnosis part 1: linear long tail hurts more than nonlinearity
11. Diagnosis part 2: more context does not help (context table — the punchline)
12. The bottleneck, and the fix it points to
13. Demo slide
14. Honesty / what is missing + conclusion

That is 14 — feel free to merge a couple to land near 12.

Before you write the file, tell me in two or three sentences what story you are going to tell across the slides, so I can correct you if the emphasis is wrong. Then write the full `.typ`.
