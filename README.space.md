---
title: Secom Production Anomaly
emoji: 📚
colorFrom: green
colorTo: gray
sdk: docker
pinned: false
license: mit
short_description: UCI Semiconductor Anomaly in Production Model API.
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Overview

- LSTM-based anomaly detector over 10-step SECOM sensor windows (590 features)
- Default classification threshold: **0.7325** (F1-optimised)
- Validation metrics: Accuracy 96.92%, Precision 73.11%, Recall 84.47%, F1 78.38%
- `/preprocess` scales a single reading; `/predict` maintains a rolling window and returns probabilities once the buffer is filled

