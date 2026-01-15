import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from preprocess import get_dataset, convert_symbols_to_inputs, get_vocab_size
from modeling import TransformerConfig, Transformer, get_padding_mask, get_mutual_mask, get_tril_mask, get_mem_tril_mask

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_eval_dataloader(products_list, reactants_list, product_ids_list, batch_size, max_length):
    """修改后的评估数据加载器
    Args:
        products_list: 产物SMILES列表
        reactants_list: 反应物SMILES列表
        product_ids_list: 产物ID列表（新增）
        batch_size: 批次大小
        max_length: 最大序列长度
    """
    # 计算最大深度
    max_depth = max(len(products) for products in products_list)
    
    # 转换输入
    (products_input, 
     products_input_mask, 
     reactants_input, 
     reactants_input_mask, 
     memory_input_mask, 
     label_input,
     product_ids) = convert_symbols_to_inputs(  # 更新函数调用
        products_list, 
        reactants_list,
        product_ids_list,  # 新增产物ID参数
        max_depth, 
        max_length
    )
    
    # 转换为tensor并移到设备
    products_input = torch.LongTensor(products_input).to(device)
    reactants_input = torch.LongTensor(reactants_input).to(device)
    label_input = torch.LongTensor(label_input).to(device)
    products_input_mask = torch.FloatTensor(products_input_mask).to(device)
    reactants_input_mask = torch.FloatTensor(reactants_input_mask).to(device)
    memory_input_mask = torch.FloatTensor(memory_input_mask).to(device)
    product_ids = torch.LongTensor(product_ids).to(device)  # 新增
    
    # 创建数据集
    eval_data = torch.utils.data.TensorDataset(
        products_input, 
        reactants_input, 
        label_input, 
        products_input_mask, 
        reactants_input_mask, 
        memory_input_mask,
        product_ids  # 新增产物ID
    )
    
    # 创建数据加载器
    eval_loader = torch.utils.data.DataLoader(
        eval_data, 
        batch_size=batch_size
    )
    return eval_loader

def calculate_metrics(logits, label_ids, memory_mask, pad_idx):
    """
    计算 Line Accuracy (Step级) 和 Path Accuracy (路径级)
    Args:
        logits: [Batch, Depth, Seq_Len, Vocab] 或 [Batch, Depth, Seq_Len] (如果是argmax后)
        label_ids: [Batch, Depth, Seq_Len]
        memory_mask: [Batch, Depth] (1代表有效步骤，0代表padding步骤)
        pad_idx: int
    Returns:
        step_acc_count: 预测正确的总步数
        total_steps: 总有效步数
        path_acc_count: 预测正确的完整路径数
        total_paths: 总路径数 (Batch Size)
        token_acc_val: token准确率
    """
    # 1. 获取预测结果
    if len(logits.shape) == 4:
        pred_ids = logits.argmax(dim=-1) # [B, D, L]
    else:
        pred_ids = logits

    # 2. 确保维度对齐
    batch_size, max_depth, seq_len = label_ids.shape
    
    # 3. Token 级比对矩阵 [B, D, L]
    # 预测正确 且 标签不是pad
    token_match = (pred_ids == label_ids)
    
    # 忽略 padding 位置的比较 (label为pad的地方视为自动正确，或者只看label非pad的地方)
    # 方法：如果 label 是 pad，那对应位置算 True (不影响这一行的最终结果)
    is_pad = (label_ids == pad_idx)
    token_match_valid = token_match | is_pad 
    
    # 4. Step 级准确率 (Line Accuracy)
    # 这一步的所有 token 都必须匹配 [B, D]
    step_is_correct = token_match_valid.all(dim=-1)
    
    # === 核心：结合 memory_mask 过滤无效深度 ===
    # memory_mask 为 0 的地方（padding step），我们不关心它预测得对不对
    # 但为了计算 Path Acc，我们假设 Padding step 是"对"的，这样不会拖累 AND 运算
    memory_mask_bool = memory_mask.bool() # [B, D]
    
    # 真正的有效步骤预测正确：步骤正确 AND 是有效步骤
    valid_step_correct = step_is_correct & memory_mask_bool
    
    # 统计 Step Acc
    step_acc_count = valid_step_correct.sum().item()
    total_steps = memory_mask_bool.sum().item()
    
    # 5. Path 级准确率 (Path Accuracy)
    # 一条路径正确 = 该路径下所有有效步骤都正确
    # 逻辑：对于每一个样本，(步骤正确 OR 步骤是Padding) 必须全为 True
    path_is_correct = (step_is_correct | (~memory_mask_bool)).all(dim=-1) # [B]
    
    path_acc_count = path_is_correct.sum().item()
    total_paths = batch_size

    # 6. 计算 Token Acc (辅助)
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

# ================= 修改后的 evaluate 函数 =================
def evaluate(model, dataloader, product_features, pad_idx):
    model.eval()
    
    # 累加器
    agg_metrics = {
        "loss": 0.0,
        "step_correct": 0, "total_steps": 0,
        "path_correct": 0, "total_paths": 0,
        "token_correct": 0, "total_tokens": 0
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            products_ids, reactants_ids, label_ids, products_mask, reactants_mask, memory_mask, products_index = batch
            
            # 准备数据
            product_features_batch = product_features[products_index.cpu()].to(device)
            
            # 注意：evaluate 里的 memory_mask 可能是原始的 [B, D]，也可能是 dataloader 处理过的
            # 查看 convert_symbols_to_inputs，memory_input_mask 是 [B, D]
            # 但是 get_eval_dataloader 里转成了 tensor。
            # 关键：这里我们需要原始的 [B, D] mask 来做统计，
            # 而传给 model 的 memory_mask 需要经过 get_mem_tril_mask 处理
            
            raw_memory_mask = memory_mask.clone() # [B, D] 用于统计
            
            # 模型输入所需的 mask 处理
            mutual_mask_in = get_mutual_mask([reactants_mask, products_mask])
            products_mask_in = get_padding_mask(products_mask)
            reactants_mask_in = get_tril_mask(reactants_mask)
            memory_mask_in = get_mem_tril_mask(memory_mask) # 变成了 [B, 1, D, D] 或类似
            
            # 前向传播
            logits = model(
                products_ids, reactants_ids, 
                products_mask_in, reactants_mask_in, mutual_mask_in, memory_mask_in,
                product_features_batch
            )
            
            # 计算 Loss (展平处理)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                label_ids.reshape(-1),
                ignore_index=pad_idx,
                reduction='sum'
            )
            
            # 计算各项指标
            batch_metrics = calculate_metrics(logits, label_ids, raw_memory_mask, pad_idx)
            
            # 更新累加
            agg_metrics["loss"] += loss.item()
            for k in batch_metrics:
                if k in agg_metrics:
                    agg_metrics[k] += batch_metrics[k]

    # 计算最终平均值
    avg_loss = agg_metrics["loss"] / agg_metrics["total_tokens"] if agg_metrics["total_tokens"] > 0 else 0
    
    step_acc = agg_metrics["step_correct"] / agg_metrics["total_steps"] if agg_metrics["total_steps"] > 0 else 0
    path_acc = agg_metrics["path_correct"] / agg_metrics["total_paths"] if agg_metrics["total_paths"] > 0 else 0
    token_acc = agg_metrics["token_correct"] / agg_metrics["total_tokens"] if agg_metrics["total_tokens"] > 0 else 0
    
    # print(f"\nEvaluation Results:")
    # print(f"Loss: {avg_loss:.4f}")
    # print(f"Token Acc: {token_acc:.4%}")
    # print(f"Step Acc (Line Acc): {step_acc:.4%}")
    # print(f"Path Acc (Whole Route): {path_acc:.4%}  <-- 核心指标")
    
    return avg_loss, step_acc, path_acc, token_acc

def convert_dict_to_list(products_dict, reactants_dict, product_ids_dict):
    """将深度字典格式转换为列表格式"""
    products_list = []
    reactants_list = []
    product_ids_list = []  # 新增
    for depth in sorted(products_dict.keys()):
        products_list.extend(products_dict[depth])
        reactants_list.extend(reactants_dict[depth])
        product_ids_list.extend(product_ids_dict[depth])  # 新增
    return products_list, reactants_list, product_ids_list

if __name__ == "__main__":
    print("device:", device)
    # 参数
    batch_size = 32
    max_length = 200
    pad_idx = 0  # 如果你的pad不是0，请替换

    # 加载模型
    config = TransformerConfig(
        vocab_size=get_vocab_size(),
        max_length=max_length,
        embedding_size=64,
        hidden_size=640,
        num_hidden_layers=3,
        num_attention_heads=10,
        intermediate_size=512,
        hidden_dropout_prob=0.1
    )
    model = Transformer(config)
    checkpoint = torch.load("FusionRetro/models/model.pkl", map_location=device)
    if isinstance(checkpoint, torch.nn.DataParallel):
        checkpoint = checkpoint.module
    try:
        model.load_state_dict(checkpoint.state_dict(),strict=True)
    except RuntimeError as e:
        print("Error loading model state_dict:", e)
    model.to(device)
    print("Model loaded successfully.")

    # 验证集
    valid_products_dict, valid_reactants_dict, valid_product_ids_dict = get_valid_dataset()  # 更新函数调用
    valid_products_list, valid_reactants_list, valid_product_ids_list = convert_dict_to_list(
        valid_products_dict, 
        valid_reactants_dict,
        valid_product_ids_dict  # 新增
    )
    print(f"Validation set size: {len(valid_products_list)}")
    valid_loader = get_eval_dataloader(
        valid_products_list, 
        valid_reactants_list,
        valid_product_ids_list,  # 新增
        batch_size, 
        max_length
    )
    print("\nEvaluating on validation set...")
    evaluate(model, valid_loader, pad_idx)

    # 测试集
    test_products_dict, test_reactants_dict, test_product_ids_dict = get_test_dataset()  # 更新函数调用
    test_products_list, test_reactants_list, test_product_ids_list = convert_dict_to_list(
        test_products_dict, 
        test_reactants_dict,
        test_product_ids_dict  # 新增
    )
    print(f"Test set size: {len(test_products_list)}")
    test_loader = get_eval_dataloader(
        test_products_list, 
        test_reactants_list,
        test_product_ids_list,  # 新增
        batch_size, 
        max_length
    )
    print("\nEvaluating on test set...")
    evaluate(model, test_loader, pad_idx)