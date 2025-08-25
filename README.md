### DS2D – Floorplan Generation (Dataset → SFT → GRPO → Inference)

Train and evaluate large models for 2D floorplan generation from the RPLAN dataset using a two-stage recipe: Supervised Fine-Tuning (SFT) followed by GRPO, then run inference with vLLM.

- **Dataset prep**: Convert RPLAN rasters → HouseGAN++ JSONs → Hugging Face `DatasetDict`
- **Training**: 1) SFT via llama-cookbook; 2) GRPO with TRL
- **Inference**: Batched generation with vLLM (optional LoRA)

---

### 1) Dataset Preparation

This repo expects RPLAN converted to HouseGAN++ JSON first, then to a Hugging Face dataset.

- **Step 1: Download the RPLAN dataset** via the official request form: https://docs.google.com/forms/d/e/1FAIpQLSfwteilXzURRKDI5QopWCyOGkeb_CFFbRwtQ0SOPhEg0KGSfw/viewform

- **Step 2: Convert rasters to JSON with the HouseGAN data reader** https://github.com/sepidsh/Housegan-data-reader

Resulting folder structure:

```text
datasets/
  rplan_json/               
```

- **Step 3: Convert HouseGAN JSONs to HF `DatasetDict`**

```bash
python src/dataset_convert/rplan.py
```
Resulting folder structure:

```text
datasets/
  rplan_5/                     
    train/
    validation/
    test/
  rplan_6/
    train/
    validation/
    test/
  rplan_7/
    train/
    validation/
    test/
  rplan_8/
    train/
    validation/
    test/
```

---

### 2) Two-Step Training

#### A) Supervised Fine-Tuning (SFT) via llama-cookbook

- Multi-node (Slurm) example:

Arguments: EPOCHS ROOM_NUMBER
```bash
sbatch scripts/train_multi_node.slurm 2 6
```
- Adjust model name, epochs, and dataset path inside the Slurm script as needed.

#### B) GRPO (Reinforcement, TRL)

Finetune with rewards computed from total area and graph compatibility.

Arguments: ROOM_NUMBER
```bash
sbatch scripts/grpo_multi_node.slurm 6
```
Flags of interest:
- `--model`: base or SFT checkpoint
- `--dataset`: dataset folder 


Note: Rewards implemented in `src/grpo/reward_calculator.py`.

---

### 3) Inference (vLLM)

Run batched generation with optional LoRA:
```bash
python src/pred/run_generation.py \
  --batch_size 64 \
  --model_name_or_path models/Llama-3.3-70B-Instruct \
  --lora_adapter_path output/sft_rplan6  # optional (PEFT adapter) \
  --dataset_name_or_path datasets/final_2/rplan_6 \
  --output_dir results/rplan6_demo \
  --test_range "1,100" \
  --use_sampling
```
Outputs per sample are stored under `results/<exp>/<index>/` with `0.json` (prediction), `prompt.json` (input), and `analysis/sample.json` (original sample).

---

### References

- RPLAN request form: https://docs.google.com/forms/d/e/1FAIpQLSfwteilXzURRKDI5QopWCyOGkeb_CFFbRwtQ0SOPhEg0KGSfw/viewform
- HouseGAN data reader for RPLAN raster → JSON conversion: https://github.com/sepidsh/Housegan-data-reader 