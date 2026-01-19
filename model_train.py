import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import random
import numpy as np
import math
import datetime
import torch
import torch.nn as nn
from tqdm import tqdm, trange
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from preprocess import get_dataset_all_feature, convert_symbols_to_inputs, get_vocab_size
from modeling import TransformerConfig, Transformer, get_padding_mask, get_mutual_mask, get_tril_mask, get_mem_tril_mask
from model_eval import evaluate, get_eval_dataloader, convert_dict_to_list

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.info')


def load_product_features(checkpoint_path):
    print(f"Loading product features from {checkpoint_path}")
    
    if checkpoint_path.endswith('.pt'):
        checkpoint = torch.load(checkpoint_path)
        if 'embedding' in checkpoint:
            features = checkpoint['embedding']
        else:
            raise KeyError(f"No 'embedding' key found in checkpoint {checkpoint_path}")
            
    elif checkpoint_path.endswith('.npy'):
        features = np.load(checkpoint_path)
        features = torch.FloatTensor(features)
    else:
        raise ValueError(f"Unsupported file format. Expected .pt or .npy, got: {checkpoint_path}")
    
    if isinstance(features, np.ndarray):
        features = torch.FloatTensor(features)
    
    feature_dim = features.shape[1]
    pad_vector = torch.zeros(1, feature_dim) 
    features = torch.cat([features, pad_vector], dim=0)

    print(f"feat shape: {features.shape}")
    return features.float()

def get_grad_norm(model, norm_type=2):
    parameters = [p for p in model.parameters() if p.grad is not None]
    if len(parameters) == 0:
        return 0.0
    
    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
        norm_type
    )
    return total_norm.item()

# ==================== parser & path setting ====================
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--max_length', type=int, default=200, help='The max length of a molecule.')
parser.add_argument('--max_depth', type=int, default=10, help='The max depth of a synthesis route.')
parser.add_argument('--embedding_size', type=int, default=64, help='The size of embeddings')
parser.add_argument('--hidden_size', type=int, default=640, help='The size of hidden units')
parser.add_argument('--num_hidden_layers', type=int, default=3, help='Number of layers in encoder\'s module. Default 3.')
parser.add_argument('--num_attention_heads', type=int, default=10, help='Number of attention heads. Default 10.')
parser.add_argument('--intermediate_size', type=int, default=512, help='The size of hidden units of position-wise layer.')
parser.add_argument('--hidden_dropout_prob', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument("--batch_size", default=32, type=int, help="Total batch size for training.")
parser.add_argument('--finetune_lr', type=float, default=1e-4, help="微调初始学习率")
parser.add_argument('--pretrained_path', type=str, default="models/model.pkl", help='Pretrained model path')
parser.add_argument('--max_grad_norm', type=float, default=0.0, help='梯度裁剪阈值，0表示不裁剪')
parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing epsilon (default: 0.1)')

args = parser.parse_args()
print(args.epochs, args.batch_size, args.pretrained_path)
print(f"Label Smoothing: {args.label_smoothing}")
print(f"max_grad_norm: {args.max_grad_norm}")

entity_file = 'Data/Train/for_embedding/all_molecules_clean.txt'
features_path = 'rgcn/global_emb_FP_512/embedding.npy'

train_data_file = 'Data/Train/for_model/clean_train_FINAL.json'
valid_data_file = 'Data/Train/valid_canolize_dataset.json'

ckpt_dir = "models"
os.makedirs(ckpt_dir, exist_ok=True)
cache_file = ckpt_dir + "/fp_cache.npz"

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpu = torch.cuda.device_count()

config = TransformerConfig(vocab_size=get_vocab_size(),
                           max_length=args.max_length,
                           embedding_size=args.embedding_size,
                           hidden_size=args.hidden_size,
                           num_hidden_layers=args.num_hidden_layers,
                           num_attention_heads=args.num_attention_heads,
                           intermediate_size=args.intermediate_size,
                           hidden_dropout_prob=args.hidden_dropout_prob)


# === Get val data dataloader ===
valid_products_dict, valid_reactants_dict, valid_product_ids_dict = get_dataset_all_feature(
    valid_data_file,
    entity_file, 
    features_path, 
    cache_file
)

valid_products_list, valid_reactants_list, valid_product_ids_list = convert_dict_to_list(
    valid_products_dict, valid_reactants_dict, valid_product_ids_dict
)
valid_loader = get_eval_dataloader(
    valid_products_list, valid_reactants_list, valid_product_ids_list, 
    args.batch_size, args.max_length
)

#  === Get train data dataloader ===
depth_products_list, depth_reactants_list, depth_product_ids_list = get_dataset_all_feature(
    train_data_file,
    entity_file, 
    features_path, 
    cache_file
)

def get_depth_dataloader(depth):
    (train_products_input, 
     train_products_input_mask, 
     train_reactants_input, 
     train_reactants_input_mask, 
     train_memory_input_mask, 
     train_label_input,
     train_product_ids) = convert_symbols_to_inputs(
        depth_products_list[depth], 
        depth_reactants_list[depth],
        depth_product_ids_list[depth],
        depth, 
        args.max_length
    )

    train_products_input = torch.LongTensor(train_products_input).to(device)
    train_reactants_input = torch.LongTensor(train_reactants_input).to(device)
    train_label_input = torch.LongTensor(train_label_input).to(device)
    train_products_input_mask = torch.FloatTensor(train_products_input_mask).to(device)
    train_reactants_input_mask = torch.FloatTensor(train_reactants_input_mask).to(device)
    train_memory_input_mask = torch.FloatTensor(train_memory_input_mask).to(device)
    train_product_ids = torch.LongTensor(train_product_ids).to(device)

    train_data = TensorDataset(
        train_products_input, 
        train_reactants_input, 
        train_label_input, 
        train_products_input_mask, 
        train_reactants_input_mask, 
        train_memory_input_mask,
        train_product_ids
    )

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, 
        sampler=train_sampler, 
        batch_size=args.batch_size
    )
    return train_dataloader

train_dataloader_list = []
for depth in list(depth_products_list.keys()):
    train_dataloader_list.append(get_depth_dataloader(depth))


mean_pooling_features_path = ckpt_dir + '/fp_cache_custom_features.npy'
try:
    product_features = load_product_features(mean_pooling_features_path)
    print("Product mean_pooling_features loaded with shape:", product_features.shape)
    print("load from:", mean_pooling_features_path)
except Exception as e:
    print(f"Error loading product features: {e}")
    raise



# ==================== Model ====================
model = Transformer(config)
checkpoint = torch.load(args.pretrained_path)
if isinstance(checkpoint, torch.nn.DataParallel):
    checkpoint = checkpoint.module

model_dict = model.state_dict()
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    pretrained_dict = checkpoint['model_state_dict']
elif hasattr(checkpoint, 'state_dict'):
    pretrained_dict = checkpoint.state_dict()
else:
    raise ValueError("Unsupported checkpoint format")

matched_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
model_dict.update(matched_dict)
model.load_state_dict(model_dict, strict=False)

total_params = len(model_dict)
loaded_params = len(matched_dict)
print(f"loaded {loaded_params}/{total_params} parameters from pretrained model.")
print(f"randomly initialized new parameters.")

if num_gpu > 1:
    model = torch.nn.DataParallel(model)
model.to(device)

# ==================== Optimizer and Scheduler ====================
optimizer = optim.Adam(model.parameters(), lr=args.finetune_lr)

def get_cosine_with_warmup_scheduler(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / args.finetune_lr, cosine_decay)
    return LambdaLR(optimizer, lr_lambda)


num_training_steps = sum(len(dl) for dl in train_dataloader_list) * args.epochs
num_warmup_steps = int(0.1 * num_training_steps)

scheduler = get_cosine_with_warmup_scheduler(
    optimizer, 
    num_warmup_steps=num_warmup_steps, 
    num_training_steps=num_training_steps,
    min_lr=1e-6
)

criterion = nn.CrossEntropyLoss(
    label_smoothing=args.label_smoothing,
    ignore_index=0,
    reduction='mean'
)
print(f"Loss Function Initialized: nn.CrossEntropyLoss(label_smoothing={args.label_smoothing})")

# ==================== Training Setup ====================
continue_epoch = 0
global_step = 0

best_epoch = 0
best_step_acc = 0.0
best_path_acc = 0.0
best_step_epoch = 0
best_path_epoch = 0
best_both_epoch = 0

log_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = ckpt_dir + f"/training_log_{log_time}.txt"


# ==================== log ====================
def log_metrics(epoch: int, metrics: dict, first: bool = False):
    mode = 'w' if first else 'a'
    with open(log_file, mode, encoding='utf-8') as f:
        if first:
            f.write(f"Start: {datetime.datetime.now()}\n")
            f.write(f"Args: {vars(args)}\n")
            f.write(f"Total steps ≈ {num_training_steps} | Warmup: {num_warmup_steps}\n")
            f.write(f"Label Smoothing: {args.label_smoothing}\n")
            f.write(f"Max Grad Norm: {args.max_grad_norm} (0=disabled)\n")
            f.write("=" * 80 + "\n")
        
        f.write(f"[Epoch {epoch:3d}] ")
        for k, v in metrics.items():
            if isinstance(v, dict):
                f.write(" | " + " ".join(f"{sk}:{sv:.5f}" for sk, sv in v.items()))
            else:
                f.write(f" | {k}:{v:.6f}")
        f.write("\n")

print(f"Starting training on device: {device}")
print(f"Number of GPUs: {num_gpu}")

# ==================== Training Loop ====================
epochs_no_improve = 0
try:
    for epoch in trange(continue_epoch + 1, continue_epoch + int(args.epochs) + 1, desc="Epoch"):
        model.train()
        total_t = 0
        total_sum_loss = 0
        depth_loss = {}
        depth = 2
        grad_norms = []
        
        for train_dataloader in train_dataloader_list:
            t = 0
            sum_loss = 0
            for step, batch in enumerate(train_dataloader):
                optimizer.zero_grad(set_to_none=True)
                products_ids, reactants_ids, label_ids, products_mask, reactants_mask, memory_mask, products_index = batch
                batch_size, max_depth = products_index.shape

                product_features_batch = product_features[products_index.cpu()].to(device)

                mutual_mask = get_mutual_mask([reactants_mask, products_mask])
                products_mask = get_padding_mask(products_mask)
                reactants_mask = get_tril_mask(reactants_mask)
                memory_mask = get_mem_tril_mask(memory_mask)
                
                logits = model(
                    products_ids, reactants_ids, 
                    products_mask, reactants_mask, 
                    mutual_mask, memory_mask,
                    product_features_batch
                )
                
                logits_flat = torch.reshape(logits, (-1, logits.shape[-1]))
                labels_flat = torch.flatten(label_ids)
                
                loss = criterion(logits_flat, labels_flat)
                loss.backward()

                grad_norm = get_grad_norm(model)
                grad_norms.append(grad_norm)
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                global_step += 1

                current_t = labels_flat.size()[0]
                current_sum_loss = loss.item() * current_t
                total_sum_loss += current_sum_loss
                total_t += current_t
                sum_loss += current_sum_loss
                t += current_t
                
            torch.cuda.empty_cache()
            depth += 1

        avg_loss = total_sum_loss / total_t
        
        grad_norms_np = np.array(grad_norms)
        grad_stats = {
            'mean': float(np.mean(grad_norms_np)),
            'max': float(np.max(grad_norms_np)),
            'p95': float(np.percentile(grad_norms_np, 95))
        }
        
        # evalulate
        if epoch % 10 == 0 or epoch >= args.epochs - 100 or epoch == 1:
            model.eval()
            with torch.no_grad():
                valid_loss, step_acc, path_acc, token_acc = evaluate(
                    model, valid_loader, product_features, pad_idx=0
                )

            current_lrs = [pg['lr'] for pg in optimizer.param_groups]

            metrics = {
                'train_loss': avg_loss,
                'test': {
                    'loss': valid_loss, 
                    'step_acc': step_acc, 
                    'path_acc': path_acc, 
                    'token_acc': token_acc
                },
                'lr': current_lrs[0],
                'grad': grad_stats
            }
            log_metrics(epoch, metrics, first=(epoch == 1))
            

            path_improved = path_acc > best_path_acc
            step_improved = step_acc > best_step_acc
            if path_improved and step_improved:
                best_path_acc = path_acc
                best_step_acc = step_acc
                best_both_epoch = epoch
                save_path = f"{ckpt_dir}/finetune_best_model_ffn_gate_both.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if num_gpu > 1 else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'path_acc': path_acc,
                    'step_acc': step_acc,
                    'label_smoothing': args.label_smoothing,
                    'grad_stats': grad_stats
                }, save_path)
                print(f"New best (both) → path_acc: {path_acc:.4f}, step_acc: {step_acc:.4f} saved!")
            else:
                if path_improved:
                    best_path_acc = path_acc
                    best_path_epoch = epoch
                    save_path = f"{ckpt_dir}/finetune_best_model_ffn_gate_path.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict() if num_gpu > 1 else model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'path_acc': path_acc,
                        'step_acc': step_acc,
                        'grad_stats': grad_stats
                    }, save_path)
                    print(f"New path best → {path_acc:.4f} saved!")

                if step_improved:
                    best_step_acc = step_acc
                    best_step_epoch = epoch
                    save_path = f"{ckpt_dir}/finetune_best_model_ffn_gate_step.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict() if num_gpu > 1 else model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'path_acc': path_acc,
                        'step_acc': step_acc,
                        'grad_stats': grad_stats
                    }, save_path)
                    print(f"New step best → {step_acc:.4f} saved!")

        else:
            metrics = {
                'train_loss': avg_loss, 
                'lr': optimizer.param_groups[0]['lr'],
                'grad': grad_stats
            }
            log_metrics(epoch, metrics)

except KeyboardInterrupt:
    print("Training interrupted by user")
except Exception as e:
    print(f"Error occurred: {str(e)}")
    raise e
finally:
    # ==================== training summary ====================
    end_time = datetime.datetime.now()
    
    summary_lines = [
        "",
        "=" * 80,
        "TRAINING SUMMARY",
        "=" * 80,
        f"End time: {end_time}",
        f"Total epochs: {epoch if 'epoch' in dir() else 'N/A'}",
        f"Total steps: {global_step}",
        "",
        "--- Best Results ---",
        f"Best path Acc: {best_path_acc:.4f} @ epoch {best_path_epoch}",
        f"Best step Acc: {best_step_acc:.4f} @ epoch {best_step_epoch}",
        f"Best both @ epoch {best_both_epoch}",
        "",
        "--- Hyperparameters ---",
        f"Batch size: {args.batch_size}",
        f"Learning rate: {args.finetune_lr}",
        f"Label smoothing: {args.label_smoothing}",
        f"Max grad norm: {args.max_grad_norm} {'(disabled)' if args.max_grad_norm == 0 else ''}",
        f"Warmup steps: {num_warmup_steps}",
        "",
        "--- Model ---",
        f"Pretrained path: {args.pretrained_path}",
        f"Save directory: {ckpt_dir}",
        "=" * 80
    ]
    
    for line in summary_lines:
        print(line)
    # ==================== write summary to log file ====================
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            for line in summary_lines:
                f.write(line + "\n")
        print(f"\nSummary appended to: {log_file}")
    except Exception as e:

        print(f"Warning: Could not write summary to log file: {e}")
