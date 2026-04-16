# Stage 1 Progress Summary

## Background

As part of my additional training in compact objects and gravitational-wave data analysis, I worked on simulated LISA data related to a neutron-star–white-dwarf (NS-WD) binary system.

## What I completed

At the current stage, I completed the following tasks:

- Inspected the time-domain structure of the simulated strain signal
- Performed FFT-based spectral analysis and low-frequency zoom-in checks
- Compared different window functions and examined spectral leakage
- Used Welch PSD to estimate the noise power spectral density
- Generated STFT spectrograms to study time-frequency structure
- Constructed matched-filter pipelines, including noise-weighted filtering
- Marked peak positions and diagnosed edge effects
- Performed EDGE parameter scans to test robustness
- Ran Monte Carlo noise-only trials to estimate peak distributions and p-values
- Performed injection tests to study recovery behavior and localization error
- Refined template-start scans to improve alignment between template and data

## Current understanding

The current results suggest that the apparent candidate peak is sensitive to edge treatment and noise modeling. Under more robust settings, the peak does not significantly exceed the tail of the noise-only distribution, so the current stage is not sufficient to claim a statistically significant detection.

## Repository contents

This repository includes:
- summary documents
- representative figures
- analysis scripts for FFT, STFT, PSD estimation, matched filtering, Monte Carlo tests, injection tests, and refinement scans

## Note

This repository is mainly for academic communication and research training display.
