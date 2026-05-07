# TrafficSTLLM — Spatio-Temporal Traffic Prediction for Kolkata

---

## What this is

Kolkata's traffic doesn't follow a pattern you can fix with a timer. Congestion is constant, sensors go offline without warning, and existing deep learning models for traffic prediction have mostly been built on Western cities or synthetic simulation data. This thesis tries to do something about that.

TrafficSTLLM is a hybrid neural architecture that jointly does two things at once — it predicts traffic speeds 15 minutes into the future across 17 sensor locations, and simultaneously reconstructs the readings at sensors that have gone offline. Both tasks share the same model, the same training pass, and the same learned representations.

The model couples a Graph Attention Network (which handles the spatial relationships between road segments) with a LoRA-adapted GPT-2 (which handles the temporal sequence modelling). It was trained on what is, to our knowledge, the first minute-resolution real-sensor traffic dataset collected for any Indian city — pulled from the TomTom Flow Segment API across Kolkata's northern corridors, central business district, Eastern Metropolitan Bypass, and southern belt, from February to March 2026.

TrafficSTLLM_v2 extends this with a Gaussian Mixture Model that discovers 7 recurring traffic regimes from the data (morning rush, evening peak, late-night quiet, etc.) and uses those as soft supervision signals during training via KL-divergence regularisation.

---

## Results

**Temporal forecasting — 15 minutes ahead, all 17 nodes:**

| Masking type | MAE | R² |
|---|---|---|
| Arbitrary (30–40% random) | 0.311 km/h | 0.9857 |
| Fixed (permanent outage) | 0.486 km/h | 0.9546 |

**Spatial imputation — reconstructing offline sensors:**

| Masking type | MAE | R² |
|---|---|---|
| Arbitrary | 0.809 km/h | 0.9653 |
| Fixed | 1.687 km/h | 0.8977 |

The stress test showed that reliable imputation holds up to 11 of 17 sensors being offline simultaneously — that's 65% sensor failure — before accuracy falls below operational thresholds. For a city with limited sensor maintenance budgets, that number matters.

**Traffic period classification (TrafficSTLLM_v2):**

| Branch | Accuracy | Macro-F1 | Latency |
|---|---|---|---|
| Branch A — end-to-end LLM head | 77.81% | 75.78% | 122.34 ms/batch |
| Branch B — standalone MLP | 88.38% | 79.17% | 0.41 ms/batch |

Branch B is the one worth deploying. It's 299x faster and actually more accurate — turns out traffic period classification is mostly a function of time-of-day and average speed, not the full sequence context.

---

## Architecture

The base model (TrafficSTLLM) has four stages.

The input is a 30-minute window of speed, travel time, and confidence readings across all 17 sensors. Before encoding, 30–40% of sensor nodes are randomly zeroed out to simulate sensor dropout — this is what forces the model to learn meaningful spatial inference rather than just reading each sensor independently.

The Graph Attention Network (two layers, residual skip connection) encodes spatial relationships using road-network edge weights derived from OpenStreetMap shortest-path distances. Each node attends to its neighbours with learned weights, so the model figures out which nearby sensors are actually informative for predicting a given location.

The GAT embeddings are then projected into GPT-2's token space — one token per timestep, 30 tokens total. GPT-2 is kept frozen; only the LoRA adapters (rank=8, applied to the Q and V attention projections) are trained. This brings the trainable parameter count down to about 2.2 million out of GPT-2's 124 million.

Two output heads sit on top. The temporal head forecasts speeds at all 17 nodes for the next 15 minutes. The spatial head reconstructs the masked sensor readings at the current timestep. Both are trained jointly with a combined loss — a speed-aware Huber loss for forecasting and MSE for imputation, balanced at λ=0.5.

TrafficSTLLM_v2 adds cyclic time-of-day and day-of-week embeddings, a GMM-based period classifier head with KL-divergence regularisation, and the lightweight Branch B MLP for deployment.

---

## Data

Collected via the TomTom Flow Segment Data API v4, queried every 60 seconds at zoom level 20 (segment-level resolution). Three features per sensor per timestep: current speed (km/h), current travel time (seconds), and a confidence score.

17 sensor locations were chosen to cover four urban zones — northern corridor, central business district, Eastern Metropolitan Bypass, and southern residential-commercial belt. The adjacency graph connects sensors within 5 km shortest-path distance on the actual OSM road network, with edge weights as inverse distances.

The preprocessing pipeline handles timestamp alignment, deduplication, reindexing to a strict 1-minute grid, linear interpolation for short gaps, imputation flagging, feature normalisation, and outlier clipping — in that order.

Dataset splits use a sliding window of H=30 input steps and τ=15 output steps:

| Split | Sequences |
|---|---|
| Train | 23,839 |
| Validation | 5,109 |
| Test | 5,109 |

Raw data is not included here due to TomTom API licensing. The collection and preprocessing scripts are fully provided — you'll need your own API key.

---

## Repo Structure

```
data/
    collect.py              # TomTom API collection class
    preprocess.py           # 7-step preprocessing pipeline
    graph_construction.py   # OSMnx road graph builder
    gmm_clustering.py       # GMM period discovery

models/
    gat_encoder.py          # Graph Attention Network
    lora_gpt2.py            # LoRA-adapted GPT-2
    traffic_stllm.py        # Base model
    traffic_stllm_v2.py     # GMM-extended model

train/
    train.py                # Main training script
    masking.py              # Arbitrary and fixed masking
    incremental_update.py   # Checkpoint-based fine-tuning

eval/
    evaluate.py             # Forecasting and imputation metrics
    stress_test.py          # Saturation analysis

notebooks/
    eda.ipynb
    results_forecasting.ipynb
    results_imputation.ipynb
```

---

## Incremental Updates

When new traffic data comes in, there's no need to retrain from scratch. The existing checkpoint is loaded, all scaler objects are reused without refitting (to preserve the original normalisation), and the model is fine-tuned for 10 epochs at a learning rate of 1e-5. That's five times lower than the spatial encoder's initial rate and ten times lower than the GPT-2 LoRA rate — conservative enough to adapt to distributional drift in the new data without overwriting what the model learned on the full training set.

```bash
python train/incremental_update.py \
  --checkpoint checkpoints/best_model.pt \
  --new_data data/new_observations.csv \
  --epochs 10 \
  --lr 1e-5
```

---

## Training Details

| Parameter | Value |
|---|---|
| Framework | PyTorch + PyTorch Geometric |
| GPT-2 backbone | HuggingFace GPT-2 (124M params) |
| LoRA rank | 8 (Q and V projections only) |
| Trainable parameters | ~2.2M |
| Optimiser | AdamW, weight decay 1e-4 |
| LR schedule | Linear warmup + cosine decay |
| Batch size | 64 sequences |
| Hardware | NVIDIA GPU, FP16 mixed precision |

Training uses a three-phase incremental fine-tuning protocol: output heads first (GPT-2 frozen), then LoRA adapters unfrozen, then the top two GAT layers unfrozen for end-to-end training. Early stopping on validation loss, best epoch checkpointed.

---

## The 7 Traffic Periods

These were discovered entirely from data using GMM with BIC elbow selection at K=7. No manual labels.

| Period | Approximate hours |
|---|---|
| Late-Night Quiescence | 00–05h |
| Early Morning | 05–07h |
| Morning Rush Hour | 07–10h |
| Mid-Day Moderate | 10–13h |
| Early Afternoon Lull | 13–16h |
| Evening Peak | 16–20h |
| Night Moderate | 20–00h |

---
