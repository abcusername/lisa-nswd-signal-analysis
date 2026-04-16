# LISA NS-WD Signal Analysis: Frequency-Domain, Time-Frequency, and Matched-Filter Training

This repository records my additional research training on compact objects and gravitational-wave signal analysis under the guidance of **Shenghua Yu (NAOC)**.

The project focuses on simulated **LISA** data for a **neutron-star–white-dwarf (NS-WD) binary** and documents the workflow I completed at the current stage, including:

- frequency-domain analysis with FFT
- Welch PSD estimation
- STFT / spectrogram analysis
- noise-weighted matched filtering
- edge-effect diagnosis
- Monte Carlo significance tests
- injection tests
- coarse-to-refined template-start scans

The current stage is a structured training and methodology repository rather than a final scientific claim. In the present analysis, the candidate peak does not robustly exceed the tail of the noise-only distribution under more conservative settings, so the result is not yet sufficient to claim a statistically significant detection. :contentReference[oaicite:0]{index=0}

## Project Background

As part of my additional training in compact objects and gravitational-wave data analysis, I worked on simulated LISA data related to an NS-WD binary system. The goal was to build a basic but reasonably complete signal-analysis workflow that connects:

- time-domain inspection
- frequency-domain identification
- time-frequency structure
- matched-filter recovery
- robustness checks against noise and edge effects

## What I Have Completed

### 1. Time-domain and frequency-domain inspection

I first inspected the raw simulated data and templates in the time domain, and then carried out FFT-based spectral analysis to identify the main frequency structure of the signal.

This stage included:
- direct time-series inspection
- FFT amplitude and power-spectrum plots
- low-frequency zoom-in analysis
- simple fitting attempts for spectral behavior

### 2. Welch PSD and STFT analysis

To better characterize the noise and time-frequency behavior, I used:

- **Welch PSD** to estimate the effective noise power spectral density
- **STFT / spectrograms** to visualize time-frequency evolution

These steps were used to distinguish broad spectral trends, narrow features, and possible signal-supporting structures in the simulated data. :contentReference[oaicite:1]{index=1}

### 3. Noise-weighted matched filtering

I implemented matched-filter pipelines using the provided template and noise information.

This stage included:
- template interpolation and alignment
- noise-weighted filtering using estimated PSD
- matched-filter peak identification
- comparison between data peaks and template-related peaks

### 4. Peak marking and edge-effect checks

I examined whether the apparent peak behavior is sensitive to edge treatment.

This included:
- direct peak marking in matched-filter outputs
- explicit **EDGE** parameter scans
- comparison of peak values under different edge exclusions

These checks were important because an apparently strong peak can be partly driven by boundary artifacts rather than a robust physical recovery. 

### 5. Monte Carlo significance tests

I ran noise-only Monte Carlo tests to estimate whether the data peak significantly exceeds peaks expected from noise realizations.

This stage included:
- colored-noise synthesis from estimated PSD
- matched-filter processing of synthetic noise
- peak-distribution measurements
- p-value / significance-style evaluation

The current conclusion is that, under more robust settings, the data peak is not strong enough to support a statistically significant detection claim. 

### 6. Injection tests

I also carried out injection tests to study how well an injected signal can be recovered.

This stage included:
- controlled amplitude injections
- recovery-rate tests
- recovered-peak statistics
- timing-offset diagnostics

These tests helped evaluate whether the current pipeline can recover a signal reliably and how the recovery depends on amplitude and analysis settings. 

### 7. Template-start scans and refinement

To improve template alignment, I implemented:
- coarse scans over template start time
- refinement scans around the best coarse candidate
- additional consistency checks with noise modeling

This was used to identify the best-matching template segment and to test the stability of the matched-filter output. 

## Repository Structure

```text
lisa-nswd-signal-analysis/
├── docs/      # task summary, progress summary, and report materials
├── figures/   # representative plots from FFT, PSD, STFT, matched filtering, MC, and injections
├── src/       # analysis scripts organized by function
├── .gitignore
├── LICENSE
└── README.md
