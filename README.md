Mixture-of-Depths Router Tuning (LLaMA-1B)
==========================================

Goal: wrap a LLaMA-1B checkpoint (TinyLlama or compatible) with lightweight routers that learn to skip layers, freeze all base weights, and fine-tune only the routers. The code here keeps training fully differentiable (soft mixing) and adds a compute-budget penalty; at inference you can optionally enable hard skips for actual latency/FLOP savings.

What you get
------------
Router modules injected into selected decoder layers (every layer by default, configurable).

Base LLaMA weights frozen; only router parameters train.

Soft gating during training; optional hard-skip threshold at inference.

Compute-usage penalty to steer toward a target fraction of active layers.

HF Trainer-based training loop with a tiny preprocessing pipeline for text datasets.


1) Install deps and train routers on a small text set (example uses wikitext-2-raw-v1):
```
python train_router.py \
  --model_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
  --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 \
  --output_dir outputs/router_tuned \
  --sequence_length 512 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --learning_rate 5e-4 \
  --num_train_epochs 1 \
  --router_lambda 0.01 \
  --target_active_ratio 0.7
```
3) Inference with hard skips (example threshold=0.5):
```python
from transformers import AutoTokenizer
from mod_router import load_with_routers, set_router_mode
tok=AutoTokenizer.from_pretrained('outputs/router_tuned')
model=load_with_routers('outputs/router_tuned', hard_threshold=0.5)
set_router_mode(model, hard_threshold=0.5)
print(tok.decode(model.generate(tok('hello', return_tensors='pt').input_ids, max_new_tokens=30)[0]))
```

Design notes
------------
- Router = small MLP producing token-level gates (shape `[B, T, 1]`). Gate mixes the layer output with the skip path: `out = gate * layer(x) + (1 - gate) * x`.
- Compute penalty approximates expected active-layer fraction: mean gate over tokens/layers vs `target_active_ratio`. Loss = LM loss + `router_lambda * max(avg_usage - target, 0)`; set `router_lambda=0` for warm-up.
- Layer selection: by default every decoder layer gets a router; `--router_pattern every_other` only wraps every second layer; `--router_start` / `--router_stop` bound the range and you can lock first/last layers via args.
- Inference: training is soft (no speedup), inference can enable `hard_threshold` so low-gate layers skip compute. Threshold tuning controls the accuracy/speed trade-off.

