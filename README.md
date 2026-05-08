# GuideRetro: Toward Synthesizability-Aware Multi-Step Retrosynthetic Planning


## Requirements

All required packages can be installed by running:

```bash
pip install -r requirements.txt
```

Key dependencies: PyTorch 1.13.1+cu117, DGL 0.9.1+cu117, RDKit, FAISS-GPU, einops, scikit-learn.

## Data Preparation

### 1. Download Data

1. **RetroBench & Zinc Stock**: Download from [FusionRetro](https://github.com/SongtaoLiu0823/FusionRetro) and place `zinc_stock_17_04_20.hdf5` plus the RetroBench dataset files into `Data/`.
2. **Retro Building Blocks & Models**: Download [retro_data.zip](https://www.dropbox.com/scl/fi/cchn0wjz8j0dqxhr0qrom/retro_data.zip?rlkey=kqz60ec7vx7087vg1o63nucyo&e=1&dl=0). Unzip it and merge the `Data/` and `retro_star/one_step_model/` folders into the project root.
3. **USPTO-Full**: Download from [GLN](https://github.com/Hanjun-Dai/GLN) and place the files into `Data/Train/for_embedding/`.

### 2. Directory Structure

Organize the data as follows:

```text
Data/
├── Test/
│   ├── chembl_1000.pkl
│   ├── gdb17_1000.pkl
│   ├── retro*_190.pkl
│   └── test_dataset.json
├── Train/
│   ├── for_embedding/
│   │   ├── raw_test.csv
│   │   ├── raw_train.csv
│   │   └── raw_val.csv
│   └── for_model/
│       ├── train_canolize_dataset.json
│       └── valid_canolize_dataset.json
├── zinc_stock_17_04_20.hdf5
└── origin_dict.csv
```

### 3. Data Preprocessing

```bash
# Canonicalize RetroBench (train_dataset.json → train_canolize_dataset.json)
python Dataprocess/to_canolize.py --dataset train
python Dataprocess/to_canolize.py --dataset valid

# Clean training data — ensure no test molecules leak into training set
python Dataprocess/get_clear_emb_node.py
python Dataprocess/get_clear_train_data.py
```

## Project Structure

```
GuideRetro/
├── pretrain_kg_embedding_FP.py    # TransE KG embedding pretraining 
├── RGCN.py                        # RGCN training with BPR loss for molecular embeddings
├── modeling.py                    # Transformer encoder/decoder with gated fusion module
├── model_train.py                 # Single-step Transformer fine-tuning
├── preprocess.py                  # SMILES tokenization, vocab, feature extraction
├── Similar_search.py              # FAISS molecular similarity search (Morgan FP + Tanimoto)
├── retro_seach.py                 # A* multi-step retrosynthesis search
├── greedy_dfs.py                  # Greedy DFS multi-step retrosynthesis search
├── Dataprocess/                   # Data cleaning & canonicalization scripts
│   ├── to_canolize.py             # SMILES canonicalization via InChI round-trip
│   ├── get_clear_train_data.py    # Remove test-overlapping molecules from training
│   ├── get_clear_emb_node.py      # Clean embedding nodes
│   └── get_reaction_score.py      # Compute reaction scores
├── retro_star/                    # Retro* planner integration
│   ├── alg/                       # MolTree, MolNode, MolStar search algorithms
│   ├── common/                    # Shared utilities (args, fingerprints, prepare)
│   ├── model/                     # Value MLP for heuristic scoring
│   ├── retro_plan_w_guidereto.py  # Retro* planner using GuideRetro one-step model
│   ├── packages/                  # rdchiral & mlp_retrosyn (template-based baseline)
│   └──  one_step_model/                # Template-based one-step model checkpoints
├── ckpts/                         # Pretrained TransE KG embeddings
├── rgcn/                          # RGCN training outputs (embeddings, logs)
└── models/                        # Trained Transformer checkpoints
```

## Training

> **Checkpoints available:** model checkpoints are provided. Download `GuideRetro.7z` from [Google Drive](https://drive.google.com/file/d/1hcom5Ukbo_0G5BESrzVKKc9SX66bvZqv/view?usp=sharing), decompress, and place the contents under the project root to skip training.

The training pipeline has two stages:
### Stage 1.1 — Generate Packed Fingerprints

```bash
python Dataprocess/get_fp_packed.py
```

Compresses 2048-bit Morgan fingerprints into 256-byte packed arrays (8x space reduction). This is the initialization source for Stage 1.2.
**Output:** `Data/Train/for_embedding/fingerprints_packed.npy`

### Stage 1.2 — Pretrain TransE KG Embeddings
> **DGL version:** This step requires `dgl==0.4.3` (it uses the legacy `dgl.contrib.sampling.EdgeSampler` API).

```bash
DGLBACKEND=pytorch python pretrain_kg_embedding_FP.py \
    --model_name TransE_l2 \
    --dataset Embedding \
    --data_path Data/Train/for_embedding \
    --data_files all_molecules_clean.txt relations.txt clean_reactions.txt \
    --format udd_hrt \
    --batch_size 2048 \
    --neg_sample_size 128 \
    --hidden_dim 512 \
    --gamma 12.0 \
    --lr 0.1 \
    --max_step 500000 \
    --log_interval 1000 \
    --batch_size_eval 16 \
    -adv \
    --regularization_coef 1.00E-07 \
    --gpu 0 \
    --fp_path Data/Train/for_embedding/fingerprints_packed.npy
```

Learns molecular representations from the reaction knowledge graph (head -> relation -> tail triples), initialized from packed fingerprints.

**Output:** `ckpts/TransE_l2_Embedding_*_*/Embedding_TransE_l2_entity.npy`

### Stage 1.3 — Train RGCN Embeddings
> **DGL version:** This step uses `dgl==0.9.1+cu117` (the `dgl.nn.pytorch.RelGraphConv` and `dgl.dataloading.NeighborSampler` APIs).
> 
```bash
python RGCN.py \
    --hid_size 512 \
    --batch_size 3000 \
    --num_epochs 200 \
    --score_file Data/Train/for_embedding/clean_reactions_scscore.txt
```

Refines the TransE embeddings through a 2-layer Relational Graph Convolutional Network with BPR loss.

**Output:** `rgcn/global_emb_FP_512/embedding.npy` (shape: `num_molecules x 512`)


### Stage 2: Train Single-Step Transformer

Trains the encoder-decoder Transformer with feature fusion to predict reactants from products.

```bash
python model_train.py \
    --pretrained_path models/model.pkl \
    --finetune_lr 1e-4 \
    --epochs 300 \
    --batch_size 32 \
    --label_smoothing 0 \
    --max_grad_norm 0
```

**Outputs:** Best model saved to `models/` 

## Evaluation

Two evaluation modes are supported: **Exact Match**  and **Success Rate**.
### Exact Match Results

```bash
# Greedy DFS
python greedy_dfs.py --beam_size 5 --temperature 2.2

# Retro* search (A* with value function)
python retro_seach.py --use_value --beam_size 5 --temperature 2.2

# Retro*-0 search (A* without value function)
python retro_seach.py --beam_size 5 --temperature 2.2
```

### Success Rate Results 

```bash
# Retro* search (with value function)
python retro_star/retro_plan_w_guidereto.py \
    --test_routes 'Data/Test/retro*_190.pkl' --use_value_fn --temperature 1.5

# Retro*-0 search (without value function)
python retro_star/retro_plan_w_guidereto.py \
    --test_routes 'Data/Test/retro*_190.pkl' --temperature 1.5
```


## References
- GLN: [https://github.com/Hanjun-Dai/GLN](https://github.com/Hanjun-Dai/GLN)
- retro_star: [https://github.com/binghong-ml/retro_star](https://github.com/binghong-ml/retro_star)
- FusionRetro: [https://github.com/SongtaoLiu0823/FusionRetro](https://github.com/SongtaoLiu0823/FusionRetro)
  
If you use this code in your research, please cite the GuideRetro paper.
