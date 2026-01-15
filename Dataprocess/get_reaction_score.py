#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from tqdm import tqdm
from standalone_model_numpy import SCScorer
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from functools import lru_cache

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.info')


# 全局推理函数（供缓存使用）
_apply_network = None

@lru_cache(maxsize=200000)
def ultra_fast_scscore(smi: str) -> float:
    if not smi or not smi.strip():
        return 0.0
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0.0
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=True)
    arr = np.zeros((2048,), dtype=np.float32)
    for i in fp.GetOnBits():
        arr[i] = 1.0
    return float(_apply_network(arr))


def main(molecules_file: str, reactions_file: str):
    global _apply_network

    if not os.path.exists(molecules_file):
        print(f"Error: 分子文件不存在 → {molecules_file}")
        sys.exit(1)
    if not os.path.exists(reactions_file):
        print(f"Error: 反应文件不存在 → {reactions_file}")
        sys.exit(1)

    out_file = os.path.splitext(reactions_file)[0] + "_scscore.txt"

    print(f"分子文件   : {molecules_file}")
    print(f"反应文件   : {reactions_file}")
    print(f"输出文件   : {out_file}")
    print("=" * 80)

    # 1. 读取分子
    print("正在读取分子列表...")
    raw_lines = []
    with open(molecules_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(maxsplit=1)
            if len(parts) < 2: continue
            raw_lines.append((parts[0], parts[1]))

    total_mols = len(raw_lines)
    print(f"共加载 {total_mols:,} 个分子\n")

    # 2. 加载模型
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.ckpt-10654.as_numpy.json.gz")
    if not os.path.exists(model_path):
        print(f"Error: 模型未找到！请放在：{model_path}")
        sys.exit(1)

    print("正在加载 SCScorer 模型...")
    model = SCScorer()
    model.restore(model_path, FP_len=2048)
    _apply_network = model.apply  # 绑定全局推理函数
    print("模型加载完成，推理引擎已启动！\n")

    # 3. 超快打分
    print("开始闪电推理（带大缓存）...")
    entity_score = {}
    fail_count = 0

    for idx_str, smiles in tqdm(raw_lines, desc="SCScore", unit="mol", ncols=100):
        try:
            idx = int(idx_str)
        except:
            fail_count += 1
            continue
        score = ultra_fast_scscore(smiles)
        entity_score[idx] = score
        if score == 0.0:
            fail_count += 1

    print(f"\n分子打分完成！")
    print(f"   总分子数     : {total_mols:,}")
    print(f"   成功计算     : {total_mols - fail_count:,}")
    print(f"   失败/补0     : {fail_count:,}")
    print(f"   缓存统计     : {ultra_fast_scscore.cache_info()}\n")

    # 4. 处理反应 + 统计 score_diff > 0 的比例
    print(f"正在生成最终文件并统计反应趋势 → {os.path.basename(out_file)}")
    processed = 0
    positive_count = 0  # score_diff > 0 的反应数

    with open(reactions_file, "r", encoding="utf-8") as fin, \
         open(out_file, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Reactions", unit="rxn", ncols=100):
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
            except:
                continue

            diff = entity_score.get(h, 0.0) - entity_score.get(t, 0.0)
            fout.write("\t".join([str(h), str(r), str(t), f"{diff:.4f}"]) + "\n")
            processed += 1
            if diff > 0:
                positive_count += 1

    # 最终统计报告
    positive_ratio = positive_count / processed * 100 if processed > 0 else 0

    print("\n" + "=" * 80)
    print("Success: 任务完成！最终统计报告")
    print("=" * 80)
    print(f"   已处理反应总数          : {processed:,}")
    print(f"   SCScore_diff > 0 的反应 : {positive_count:,}")
    print(f"   比例                   : {positive_ratio:.2f}%  ← ← ← ← ← 关键指标！")
    print(f"   输出文件               : {out_file}")
    print(f"   缓存命中情况           : {ultra_fast_scscore.cache_info()}")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python get_reaction_score.py <molecules.txt> <reactions.txt>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

    # python Dataprocess/get_reaction_score.py Data/Train/for_embedding/all_molecules.txt Data/Train/for_embedding/clean_reactions.txt