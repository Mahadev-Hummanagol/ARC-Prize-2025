# ARC Multitensor VAE Decoder --- Explanation

## Overview

This document explains the hand‑coded VAE‑style decoder used to
reconstruct ARC (Abstraction and Reasoning Corpus) grid patterns using a
**multitensor system**.

The decoder is *not trained*. Instead, symmetric random weights +
massive enumeration allow discovering ARC rules.

------------------------------------------------------------------------

## 🎯 Goal

-   Convert ARC puzzles (grids) into a **latent compressed
    representation**
-   Decode back to grid form\
-   Evaluate millions of samples → choose best reconstruction\
-   Behaves more like a **rule synthesizer** than a neural network

------------------------------------------------------------------------

## Why a "Compressor" for ARC?

ARC grids are small (5×5, 10×10) but represent high‑level
transformations:

-   symmetry\
-   reflection\
-   grouping\
-   copy--move\
-   flood fill\
-   shape repetition\
-   spatial relationships

Humans see **patterns**, not pixels → this model tries to emulate that.

------------------------------------------------------------------------

## High‑level Structure

1.  Initialize symmetric multitensor weights\
2.  Build 4 decoding layers\
3.  Each layer performs:
    -   share_up\
    -   softmax\
    -   cummax\
    -   shift\
    -   direction_share\
    -   nonlinear\
    -   share_down\
4.  Final heads produce:
    -   output grid colors\
    -   x-mask\
    -   y-mask

------------------------------------------------------------------------

## Multitensor System

Custom modules:

-   **initializers** → create symmetric multitensors\
-   **layers** → operations like share_up, softmax, shift, cummax\
-   Together form a structured decoder for ARC rules

------------------------------------------------------------------------

## ARCCompressor Class Breakdown

### 1. Hyperparameters

    n_layers = 4
    share_up_dim = 20
    share_down_dim = 12
    decoding_dim = 6
    softmax_dim = 4
    cummax_dim = 6
    shift_dim = 6
    nonlinear_dim = 16

These are **multitensor channel dimensions**, NOT CNN channels.

They encode:

-   direction\
-   position\
-   symmetry\
-   color features

------------------------------------------------------------------------

### 2. channel_dim_fn

``` python
return 20 if dims[2] == 0 else 12
```

If *no direction* → 20 channels\
If *directional* → 12 channels

Creates symmetry between X and Y.

------------------------------------------------------------------------

### 3. Constructor

Creates:

-   latent multitensors\
-   decoder weights\
-   capacity controls\
-   all operations for 4 layers

Then applies:

-   `symmetrize_xy`\
-   `symmetrize_direction_sharing`

Ensuring ARC shapes behave identically under flipping.

------------------------------------------------------------------------

## 🔥 Forward Pass (Decoder)

### Step 1 --- Decode Latents

    x, KL_amounts, KL_names = layers.decode_latents(...)

Not trained → KL used only for scoring reconstruction candidates.

------------------------------------------------------------------------

## Step 2 --- Main 4‑Layer Processing Pipeline

### ⭐ share_up

Shares pixel → global information.\
Extracts objects, shape descriptors.

### ⭐ softmax

Allows discrete pattern selection.

### ⭐ cummax

Directional cumulative maximum:\
Behaves like **flood fill**, "extend to the right", etc.

### ⭐ shift

Move or replicate objects.\
Copy/shift pattern elements.

### ⭐ direction_share

Mixes X and Y direction structure.\
Critical for ARC tasks.

### ⭐ nonlinear

Adds complexity.

### ⭐ share_down

Send global info → back to pixels.

------------------------------------------------------------------------

## Step 3 --- Final Heads

### 🟩 Output grid

    output = layers.affine(...)

Color logits for each pixel.

### 🟦 Size masks

    x_mask = ...
    y_mask = ...

Allow predicting output size.

------------------------------------------------------------------------

## Step 4 --- Postprocess Masks

Ensure valid grid dimensions.

------------------------------------------------------------------------

# 🎉 Summary

-   Not trained --- random symmetric weights\
-   Behaves as a **rule enumerator**\
-   Four layers of directional + structural operations\
-   Can reproduce ARC rules:
    -   reflection\
    -   repetition\
    -   flood fill\
    -   copy/move\
    -   shape extraction\
-   Millions of forward passes → best one chosen

This is why ARC can be solved *without learning*.
