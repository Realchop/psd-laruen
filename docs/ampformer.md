# AmpFormer

A block-causal transformer for black-box amplifier modelling, written as the
experimental arm of this project. WaveNet is the established baseline; this
asks whether attention can match it under a controlled comparison.

Source: [`src/psd_laruen/models/amp_former.py`](../src/psd_laruen/models/amp_former.py)

**Short answer: it cannot, and the way it fails is specific enough to point at
the cause.** The rest of this document is the design, the numbers, and the
diagnosis.

---

## 1. The question

Amp modelling has been dominated by WaveNet-style dilated causal convolutions
since around 2019. Transformers have since displaced recurrent and
convolutional baselines almost everywhere else. The question here is narrow on
purpose:

> Can a block-causal transformer match a WaveNet at the same parameter count
> and the same receptive field?

Not "is a transformer better at audio". A single controlled comparison, where
architecture is the only thing that changes.

---

## 2. Design

### 2.1 Attention cannot run at the sample rate

One second of audio at 44.1 kHz is 44,100 positions. Full self-attention over
that is ~1.9 × 10⁹ pairs — roughly 7.8 GB of attention scores per head per
second of audio in float32, before gradients. That is not a tuning problem; it
does not fit.

So samples are grouped into fixed chunks of 64 before attention runs, which
brings a second down to 689 positions.

The grouping is a strided convolution:

```python
self.patch = nn.Conv1d(in_channels, model_dim,
                       kernel_size=chunk_size, stride=chunk_size)
```

When `stride == kernel_size`, a convolution *is* the chunking plus a learned
linear projection of each chunk — no separate reshape step is needed. The
inverse is a `ConvTranspose1d` with the same parameters.

This is the central design decision and, as section 5 argues, also the reason
the model loses.

### 2.2 Causality and a bounded context

A guitarist cannot be handed audio that depends on notes they have not played
yet, so attention is masked to the past:

```python
delta = index.unsqueeze(1) - index.unsqueeze(0)
return (delta >= 0) & (delta < window)
```

Query `i` attends to keys `i - window + 1 … i`. The mask is *sliding*, not just
causal — it also bounds how far back a layer reaches, which keeps the context
comparable to a convolution's receptive field rather than unbounded.

Reach compounds over depth exactly as it does in a dilated stack, because each
layer reads the previous layer's output:

```
receptive_field = (num_layers × (window − 1) + 1) × chunk_size
```

The window is deliberately quoted **in chunks, not samples**, so a config can
name a target context and `window_for_context()` inverts the formula to find
the window that hits it. That is what makes matching WaveNet's receptive field
exactly possible rather than approximate.

### 2.3 Rotary embeddings, chosen for streaming

Positions use RoPE rather than learned or additive embeddings. The reason is
not quality — it is that a rotary dot product depends **only on the difference
between two positions**.

That property is what makes real-time playback possible. `StreamingAmpFormer`
keeps a per-layer key/value cache and applies rotation fresh on every step
using positions local to `cache + new`. Because only differences matter, local
indices reproduce the same attention scores as training-time absolute indices,
while keeping the angles small.

The cache deliberately stores **un-rotated** keys. Caching rotated keys would
require an ever-growing absolute counter, and section 6 shows what that costs.

### 2.4 The model starts as a wire

The output projection is zero-initialised and the raw input is added back:

```python
nn.init.zeros_(self.unpatch.weight)
...
out = x + self.unpatch(self.norm(tokens).transpose(1, 2))
```

At step zero the model is an exact passthrough. It never has to learn to
reproduce the guitar signal — only the difference the amplifier makes to it.
Both architectures use this, so it is not a confound.

### 2.5 Misaligned input is refused, not padded

`forward` raises if the input length is not a multiple of `chunk_size`:

```python
if samples % self.chunk_size != 0:
    raise ValueError(...)
```

Padding on the left would shift the chunk grid by however much was padded, so
the same audio would be cut into chunks differently depending on the length of
the buffer it arrived in — one way during training, another during block-wise
inference. That produces a quiet, hard-to-attribute quality loss. An exception
is easier to notice than a silent phase shift.

`inference.render()` rounds its block and lead-in up to whole chunks for the
same reason.

---

## 3. Matching the baseline

Both models share the loss (`AmpLoss`), the optimiser and schedule
(`optim.build_optimizer`, AdamW + linear warmup + cosine decay), the seed, and
the data. A learning-rate schedule is a confound: comparing a transformer on
warmup-cosine against a convolution on a flat rate measures the schedule as
much as the model.

| model             | params  | context  | latency |
| ----------------- | ------: | -------: | ------: |
| WaveNet 46 ms     | 205,505 |  46.4 ms | 0.02 ms |
| AmpFormer 48 ms   | 207,297 |  47.9 ms | 1.45 ms |
| WaveNet 186 ms    | 243,265 | 185.7 ms | 0.02 ms |
| AmpFormer 187 ms  | 207,297 | 187.2 ms | 1.45 ms |

At the short context the two are within 1% on parameters and 3% on context. At
the long context WaveNet carries **17% more parameters** — an advantage to the
baseline, not to AmpFormer.

Latency differs structurally and cannot be tuned away: WaveNet emits a sample
as soon as it has one, while AmpFormer cannot emit a chunk until all 64 of its
samples have arrived. 1.45 ms is still well under what a player notices, so
this is a real cost but not a disqualifying one.

**Training and validation.** Trained on the Ibanez 2820, validated on the
Career SG — a different guitar, 43.7 minutes, 2,596 segments after silent ones
are dropped. A different instrument is a genuine generalisation test where a
random split of one recording is not.

---

## 4. Results

ESR on the validation guitar. Lower is better.

| task                        | context | WaveNet | AmpFormer |  gap |
| --------------------------- | ------: | ------: | --------: | ---: |
| cabinet *(linear, long tail)* |   46 ms | 0.00010 |   0.00967 |  97× |
| cabinet                       |  186 ms | 0.00004 |   0.01042 | 260× |
| preamp *(nonlinear, short)*   |   46 ms | 0.00091 |   0.00695 |   8× |
| preamp *(nonlinear, short)*   |  186 ms | 0.00022 |   0.00770 |  35× |
| metal *(preamp + cabinet)*    |   46 ms | 0.00622 |   0.01263 |   2× |
| metal                         |  186 ms | 0.00075 |   0.00565 |   8× |

`clean` is omitted: it is close to an identity mapping, both models solve it
trivially given the residual connection, and `esr_db` is floored at −50 dB by
the `1e-5` epsilon in `ESRLoss`. It separates nothing.

AmpFormer loses on every task at every context length.

> **Read these as an upper bound on quality.** Each figure is the *minimum*
> `val_esr` across all epochs, because `ModelCheckpoint(monitor="val_esr",
> mode="min", save_top_k=1)` keeps the best epoch and `EarlyStopping` also
> watches the same number. Selecting the best of several hundred noisy
> measurements is optimistic. The validation set is large (2,596 segments), so
> the bias is small — and gaps of 2× to 260× sit far outside it — but the
> absolute values are not unbiased estimates.

---

## 5. What the results say

### 5.1 Long memory costs more than nonlinearity

The targets in `amps.py` are built to isolate exactly this. Two of them are
single-stage and directly comparable:

- **`preamp`** is a noise gate, a high-pass, 35 dB of distortion and some
  shelving EQ — strongly nonlinear, but with only a few milliseconds of memory.
  The module comment says as much: the chain "isolates the distortion from the
  long tail a cabinet adds".
- **`cabinet`** is a single `Convolution` with a speaker impulse response —
  perfectly linear, with a long tail. Reproducing it means getting phase right
  sample by sample.

Against WaveNet, AmpFormer is roughly an order of magnitude further behind on
`cabinet` (97–260×) than on `preamp` (8–35×). That is what the chunk bottleneck
predicts: sample-accurate phase cannot be recovered from a per-chunk summary,
whereas a saturating transfer curve is forgiving because the residual connection
already supplies the waveform the curve acts on.

> **`metal` is an apparent counter-example.** It is `preamp` followed by the
> same `cabinet` convolution, so it carries the long tail too — yet it shows the
> *smallest* gap of all (2–8×). The likely explanation is that ESR normalises by
> target energy, and 35 dB of distortion dominates that energy, so the same
> absolute filtering error reads as a much smaller relative one. **This has not
> been verified**, and it is the one place where the story below rests on an
> untested assumption. §5.2 does not depend on it.

### 5.2 More context does not help it

Going from ~46 ms to ~186 ms of context:

| task    | WaveNet    | AmpFormer      |
| ------- | ---------: | -------------: |
| cabinet | 2.5× better | **0.9× — worse** |
| preamp  | 4.1× better | **0.9× — worse** |
| metal   | 8.3× better | 2.2× better    |

A model starved of context benefits from more of it. This one does not — on two
of three tasks it gets worse. Whatever limits AmpFormer, it is not reach.

### 5.3 The chunk is the bottleneck

Both observations point at the same place. The model compresses 64 samples into
one 64-dimensional vector, and reconstructs 64 samples from that vector through
a single `ConvTranspose1d`.

For a nonlinearity that is survivable: the residual carries the waveform, and
the correction needed within a chunk is largely determined by the input the
residual already supplies. For a linear filter with a long tail it is not:
sample-accurate phase cannot be recovered from a per-chunk summary, and no
amount of extra context adds resolution that the bottleneck has already
removed.

The information lost is *within-chunk*, and context length does not restore it.

### 5.4 The fix this points to

Stop predicting samples from chunk summaries. Predict **filter coefficients**
per chunk instead, and apply the filtering at the sample rate. Attention stays
cheap — still 689 positions per second — while the output regains sample
resolution. This is the first thing worth trying next, and the module docstring
already anticipates it as the fallback.

---

## 6. Measured: streaming is the more accurate path

`StreamingAmpFormer` is mathematically equivalent to the offline `forward`:
attention scores depend only on position *differences*, values are never
rotated, and the sliding mask is reconstructed from the same deltas. In exact
arithmetic the two produce identical output.

In float32 they diverge, and the divergence is one-sided. Measured against
`metal-ampformer_187ms` on the validation guitar:

| signal | chunks |  max abs diff | relative |
| -----: | -----: | ------------: | -------: |
|  0.1 s |     68 |      5.7e−07  |  8.7e−07 |
|  0.5 s |    344 |      5.7e−06  |  7.7e−06 |
|    2 s |  1,378 |      2.3e−05  |  2.9e−05 |
|   10 s |  6,890 |      1.7e−04  |  1.7e−04 |
|   30 s | 20,671 |      3.9e−04  |  3.8e−04 |
|   60 s | 41,343 |      1.1e−03  |  1.0e−03 |

Error grows linearly with signal length, and block size has no effect on it
(64, 512 and 4096 give the same figure). That is the signature of the *offline*
path degrading, not the streaming one: `forward` builds its RoPE cache from
absolute positions `0 … N−1`, so the angles grow without bound and float32
precision decays. Streaming keeps positions local to `cache + new`, so its
angles stay small no matter how long the signal runs.

The keeping-un-rotated-keys decision in §2.3 is therefore load-bearing, and the
streaming decoder is the numerically better of the two paths.

**This does not affect the results in §4.** Training and validation segments are
66,048 samples (1,032 chunks), where the divergence is ~2e−05 relative — roughly
five orders of magnitude below the ESR values being compared.

---

## 7. Known gaps

Stated plainly, because they are the obvious things to ask about:

- **No chunk-size ablation.** The claim in §5.3 is supported by two independent
  observations, but not isolated experimentally. Training at
  `chunk_size = 16` and `32` would test it directly: if resolution is the
  bottleneck, error should fall with chunk size at a fixed parameter count.
  This is the single most valuable missing experiment.
- **The filter-coefficient variant in §5.4 is untested.** It is a proposal.
- **Validation doubles as model selection**, as flagged in §4. A clean test
  split would need a third guitar; `dataset1` offers only ~3 minutes each of
  isolated notes, a different playing style rather than a different instrument,
  so it would change the measurement for the wrong reason.
- **One seed per configuration.** Gaps of 2× to 260× are unlikely to be seed
  noise, but this is not measured.
- **Aliasing is not reported here.** `metrics.aliasing_to_signal_ratio` exists
  and `evaluate` computes it; those numbers have not been collected across the
  grid yet.

Not a gap: undertrained AmpFormer. Both architectures ran under identical early
stopping (`monitor="val_esr"`, `patience=40`), so each trained until 40
consecutive epochs brought no improvement — neither was cut off by a budget.
Best-checkpoint epochs, AmpFormer against WaveNet:

|         |  46/48 ms |  186/187 ms |
| ------- | --------: | ----------: |
| clean   |  78 / 92  |    35 / 70  |
| cabinet | 177 / 145 |   138 / 204 |
| preamp  | 600 / 153 |   529 / 386 |
| metal   | 319 / 109 |   342 / 388 |

It splits four to four overall, but on the two tasks where the gap actually
matters — preamp and metal — AmpFormer trained longer in three of four cells,
by as much as 4×. It was not starved of epochs.

---

## 8. Code map

| file | what to look at |
| --- | --- |
| [`models/amp_former.py`](../src/psd_laruen/models/amp_former.py) | `AmpFormer`, `sliding_causal_mask`, `rope_cache`, `window_for_context`, `StreamingAmpFormer` |
| [`models/wavenet.py`](../src/psd_laruen/models/wavenet.py) | the baseline, and `StreamingWaveNet` for the mirrored streaming interface |
| [`losses.py`](../src/psd_laruen/losses.py) | `ESRLoss`, `AmpLoss` — the shared objective |
| [`optim.py`](../src/psd_laruen/optim.py) | shared AdamW + warmup-cosine, so the schedule is not a confound |
| [`data.py`](../src/psd_laruen/data.py) | segmenting, lead-in, the `MIN_RMS` silence filter |
| [`inference.py`](../src/psd_laruen/inference.py) | `render()` — block-wise inference with warm-up, chunk-aligned |
| [`configs/`](../configs/) | the four grid cells; `window` is what sets AmpFormer's context |
| [`scripts/train.sbatch`](../scripts/train.sbatch) | one grid cell, on an H200 |

Reproduce a cell:

```bash
sbatch scripts/train.sbatch metal AmpFormer configs/ampformer_187ms.json
```

Render an excerpt for listening:

```bash
uv run demo metal-ampformer_187ms "data/Carrer SG.wav" \
    -t "data/targets/CareerSG__metal.wav" --start 60 --seconds 12
```
