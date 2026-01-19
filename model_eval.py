import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from preprocess import get_dataset, convert_symbols_to_inputs, get_vocab_size
from modeling import TransformerConfig, Transformer, get_padding_mask, get_mutual_mask, get_tril_mask, get_mem_tril_mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_eval_dataloader(products_list, reactants_list, product_ids_list, batch_size, max_length):
    """
    Args:
        products_list
        reactants_list
        product_ids_list
        batch_size
        max_length
    """
    
    max_depth = max(len(products) for products in products_list)

    (products_input, 
     products_input_mask, 
     reactants_input, 
     reactants_input_mask, 
     memory_input_mask, 
     label_input,
     product_ids) = convert_symbols_to_inputs(  
        products_list, 
        reactants_list,
        product_ids_list, 
        max_depth, 
        max_length
    )
    
    
    products_input = torch.LongTensor(products_input).to(device)
    reactants_input = torch.LongTensor(reactants_input).to(device)
    label_input = torch.LongTensor(label_input).to(device)
    products_input_mask = torch.FloatTensor(products_input_mask).to(device)
    reactants_input_mask = torch.FloatTensor(reactants_input_mask).to(device)
    memory_input_mask = torch.FloatTensor(memory_input_mask).to(device)
    product_ids = torch.LongTensor(product_ids).to(device)  # 新增
    
    
    eval_data = torch.utils.data.TensorDataset(
        products_input, 
        reactants_input, 
        label_input, 
        products_input_mask, 
        reactants_input_mask, 
        memory_input_mask,
        product_ids  
    )
    
    
    eval_loader = torch.utils.data.DataLoader(
        eval_data, 
        batch_size=batch_size
    )
    return eval_loader

def calculate_metrics(logits, label_ids, memory_mask, pad_idx):
    """
    
    Args:
        logits
        label_ids
        memory_mask
        pad_idx: int
    Returns:
        step_acc_count
        total_steps
        path_acc_count
        total_paths
        token_acc_val
    """
   
    if len(logits.shape) == 4:
        pred_ids = logits.argmax(dim=-1) # [B, D, L]
    else:
        pred_ids = logits

    batch_size, max_depth, seq_len = label_ids.shape
    
    token_match = (pred_ids == label_ids)
    
    is_pad = (label_ids == pad_idx)
    token_match_valid = token_match | is_pad 
    
    step_is_correct = token_match_valid.all(dim=-1)
    memory_mask_bool = memory_mask.bool() # [B, D]
    
    valid_step_correct = step_is_correct & memory_mask_bool
    
    step_acc_count = valid_step_correct.sum().item()
    total_steps = memory_mask_bool.sum().item()
    
    path_is_correct = (step_is_correct | (~memory_mask_bool)).all(dim=-1) # [B]
    
    path_acc_count = path_is_correct.sum().item()
    total_paths = batch_size

    valid_tokens_mask = ~is_pad
    token_correct = (token_match & valid_tokens_mask).sum().item()
    total_tokens = valid_tokens_mask.sum().item()
    
    return {
        "step_correct": step_acc_count,
        "total_steps": total_steps,
        "path_correct": path_acc_count,
        "total_paths": total_paths,
        "token_correct": token_correct,
        "total_tokens": total_tokens
    }


def evaluate(model, dataloader, product_features, pad_idx):
    model.eval()
    
   
    agg_metrics = {
        "loss": 0.0,
        "step_correct": 0, "total_steps": 0,
        "path_correct": 0, "total_paths": 0,
        "token_correct": 0, "total_tokens": 0
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            products_ids, reactants_ids, label_ids, products_mask, reactants_mask, memory_mask, products_index = batch
            
           
            product_features_batch = product_features[products_index.cpu()].to(device)
            
            raw_memory_mask = memory_mask.clone() 
            mutual_mask_in = get_mutual_mask([reactants_mask, products_mask])
            products_mask_in = get_padding_mask(products_mask)
            reactants_mask_in = get_tril_mask(reactants_mask)
            memory_mask_in = get_mem_tril_mask(memory_mask)
            
            
            logits = model(
                products_ids, reactants_ids, 
                products_mask_in, reactants_mask_in, mutual_mask_in, memory_mask_in,
                product_features_batch
            )
            
            
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                label_ids.reshape(-1),
                ignore_index=pad_idx,
                reduction='sum'
            )
            
            
            batch_metrics = calculate_metrics(logits, label_ids, raw_memory_mask, pad_idx)
            
            
            agg_metrics["loss"] += loss.item()
            for k in batch_metrics:
                if k in agg_metrics:
                    agg_metrics[k] += batch_metrics[k]

    
    avg_loss = agg_metrics["loss"] / agg_metrics["total_tokens"] if agg_metrics["total_tokens"] > 0 else 0
    
    step_acc = agg_metrics["step_correct"] / agg_metrics["total_steps"] if agg_metrics["total_steps"] > 0 else 0
    path_acc = agg_metrics["path_correct"] / agg_metrics["total_paths"] if agg_metrics["total_paths"] > 0 else 0
    token_acc = agg_metrics["token_correct"] / agg_metrics["total_tokens"] if agg_metrics["total_tokens"] > 0 else 0
    
    # print(f"\nEvaluation Results:")
    # print(f"Loss: {avg_loss:.4f}")
    # print(f"Token Acc: {token_acc:.4%}")
    # print(f"Step Acc (Line Acc): {step_acc:.4%}")
    # print(f"Path Acc (Whole Route): {path_acc:.4%} ")
    
    return avg_loss, step_acc, path_acc, token_acc

def convert_dict_to_list(products_dict, reactants_dict, product_ids_dict):

    products_list = []
    reactants_list = []
    product_ids_list = []  
    for depth in sorted(products_dict.keys()):
        products_list.extend(products_dict[depth])
        reactants_list.extend(reactants_dict[depth])
        product_ids_list.extend(product_ids_dict[depth])  
    return products_list, reactants_list, product_ids_list
