import os
import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse

from dgl.dataloading import NeighborSampler, DataLoader , NodeDataLoader
from dgl.nn.pytorch import RelGraphConv
from collections import deque
from torch.cuda.amp import autocast, GradScaler


def construct_dgl_graph(args):
    entities = set()
    src_raw = []
    dst_raw = []
    etypes = []
    sc_list = []
    if args.score_file is None:
        raise ValueError("Please provide the score_file argument.")
    with open(args.score_file) as fin:
        for line in fin:
            parts = line.strip().split('\t')
            h, r, t, sc = parts
            h = int(h)
            t = int(t)
            entities.add(h)
            entities.add(t)
            src_raw.append(h)
            dst_raw.append(t)
            etypes.append(int(r))
            sc_list.append(float(sc))
    
    num_nodes = max(entities) + 1  
    graph = dgl.DGLGraph()
    graph.add_nodes(num_nodes)

    
    graph.add_edges(torch.tensor(src_raw), torch.tensor(dst_raw))  
    
    
    graph.edata[dgl.ETYPE] = torch.tensor(etypes)
    in_deg = graph.in_degrees().float()
    norm = 1.0 / torch.clamp(in_deg, min=1.0)
    graph.edata["norm"] = norm[graph.edges()[1]].unsqueeze(-1)

    # sc=product - reactant
    graph.edata['sc'] = -torch.tensor(sc_list, dtype=torch.float32)
    
    return graph

class RGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, num_rels, pretrained_emb=None):
        super().__init__()
        
        
        if pretrained_emb is not None:
            self.emb = nn.Embedding.from_pretrained(pretrained_emb, freeze=False)  
            print("Pretrained embeddings loaded for RGCN.")
        else:
            self.emb = nn.Embedding(num_nodes, h_dim)  
        # self.emb = nn.Embedding(num_nodes, h_dim)
        self.conv1 = RelGraphConv(
            h_dim,
            h_dim,
            num_rels,
            regularizer="bdd",
            num_bases=4,
            self_loop=True,
        )
        self.conv2 = RelGraphConv(
            h_dim,
            h_dim,
            num_rels,
            regularizer="bdd",
            num_bases=4,
            self_loop=True,
        )
        self.dropout = nn.Dropout(0.2)
        self.layers = [self.conv1,self.conv2]

    def forward(self, g, nids):
        x = self.emb(nids)
        # from ipdb import set_trace;set_trace()
        # print(x.shape)
        h = F.relu(self.conv1(g, x, g.edata[dgl.ETYPE], g.edata["norm"]))
        h = self.dropout(h)
        # print(h.shape)
        h = self.conv2(g, h, g.edata[dgl.ETYPE], g.edata["norm"])
        # print(h.shape)
        return self.dropout(h)
    
    def forward_full(self, blocks, feats):
        h = feats  # [input_nodes, h_dim]
        
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            block = block.to(h.device)
            h = layer(block, h, block.edata[dgl.ETYPE], block.edata["norm"])
            h = F.relu(h)
            h = self.dropout(h)
        
        return h  

class LinkPredict(nn.Module):
    def __init__(self, num_nodes, num_rels, h_dim=500, reg_param=0.01, pretrained_emb=None):
        super().__init__()
        self.rgcn = RGCN(num_nodes, h_dim, num_rels * 2,pretrained_emb=pretrained_emb)
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(
            self.w_relation, gain=nn.init.calculate_gain("relu")
        )

    def calc_score(self, embedding, triplets):
        s = embedding[triplets[:, 0]]
        r = self.w_relation[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def forward(self, g, nids):
        return self.rgcn(g, nids)

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, embed, triplets, labels):
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(embed, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss

def load_pretrained_embeddings(npy_path, device='cpu'):
    
    pretrained_emb = np.load(npy_path)
    
    pretrained_emb = torch.tensor(pretrained_emb, dtype=torch.float32, device=device)
    return pretrained_emb


def save_checkpoint(model, epoch, avg_loss, ckpt_dir, saved_checkpoints, save_embedding=False):
    
    ckpt_path = os.path.join(ckpt_dir, f"model_epoch{epoch}_{avg_loss:.5f}.pt")
    
    
    if len(saved_checkpoints) == saved_checkpoints.maxlen:
        oldest = saved_checkpoints.popleft()
        if os.path.exists(oldest):
            os.remove(oldest)
    
    
    save_dict = {
        'epoch': epoch,
        'loss': avg_loss,
        'model_state_dict': model.state_dict(),  
    }
    
    
    if save_embedding:
        save_dict['embedding'] = model.rgcn.emb.weight.detach().cpu().numpy()
    
    torch.save(save_dict, ckpt_path)
    saved_checkpoints.append(ckpt_path)
    print(f"New best model saved: {ckpt_path} (loss={avg_loss:.6f})")

def train(dataloader, device, model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()
    best_loss = float('inf')
    saved_checkpoints = deque(maxlen=1)
    
    ckpt_dir = os.path.join("rgcn", "global_emb_FP_512")
    loss_log_path = os.path.join(ckpt_dir, "training_loss.txt")
    # embedding save path
    best_emb_path = os.path.join(ckpt_dir, "embedding.npy")
    
    os.makedirs(ckpt_dir, exist_ok=True)

    epoch_bar = tqdm(range(args.num_epochs), desc="Training Progress", unit="epoch",
                     position=0, leave=True)

    for epoch in epoch_bar:
        model.train()
        epoch_loss = 0.0
        step_count = 0

        for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            blocks = [b.to(device) for b in blocks]
            input_nodes = input_nodes.to(device)

            optimizer.zero_grad()
            
            with autocast():
                
                feats = model.rgcn.emb(input_nodes)            
                embed = model.rgcn.forward_full(blocks, feats) 
                
                num_output = embed.shape[0]
                full_feat = torch.cat([embed, feats[num_output:]], dim=0)

                
                block = blocks[-1]
                src, dst = block.edges()                                 
                rel = block.edata[dgl.ETYPE]
                sc = block.edata['sc'].float()                            

                if sc.shape[0] == 0:
                    continue

                
                sort_idx = torch.argsort(sc, descending=True)
                sorted_dst = dst[sort_idx]
                
                sorted_dst_cpu = sorted_dst.detach().cpu().numpy()
                _, first_occurrence_indices_cpu = np.unique(sorted_dst_cpu, return_index=True)
                first_occurrence_indices = torch.from_numpy(first_occurrence_indices_cpu).to(device)
                
                max_global_idx = sort_idx[first_occurrence_indices]

                pos_src = src[max_global_idx]
                pos_rel = rel[max_global_idx]
                pos_dst = dst[max_global_idx]
                
                num_pos = pos_src.shape[0]
                if num_pos == 0:
                    continue

                # ------------------- neg sample -------------------
                all_indices = torch.arange(sc.shape[0], device=device)
                is_pos = torch.zeros(sc.shape[0], dtype=torch.bool, device=device)
                is_pos[max_global_idx] = True
                neg_indices = all_indices[~is_pos]
                
                if neg_indices.shape[0] == 0:
                    continue

                K = 10
                target_neg = num_pos * K
                
                if neg_indices.shape[0] >= target_neg:
                    perm = torch.randperm(neg_indices.shape[0], device=device)[:target_neg]
                else:
                    perm = torch.randint(0, neg_indices.shape[0], (target_neg,), device=device)
                sampled_neg_indices = neg_indices[perm]

                # ------------------- score -------------------
                pos_triples = torch.stack([pos_src, pos_rel, pos_dst], dim=1)
                pos_scores = model.calc_score(full_feat, pos_triples)

                neg_src = src[sampled_neg_indices]
                neg_rel = rel[sampled_neg_indices]
                neg_dst = dst[sampled_neg_indices]
                neg_triples = torch.stack([neg_src, neg_rel, neg_dst], dim=1)
                
                neg_scores_flat = model.calc_score(full_feat, neg_triples)
                sampled_neg = neg_scores_flat.view(num_pos, K)

                # ------------------- BPR Loss -------------------
                diff = pos_scores.unsqueeze(1) - sampled_neg
                bpr_loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()

           
            scaler.scale(bpr_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += bpr_loss.item()
            step_count += 1

        
        avg_loss = epoch_loss / step_count if step_count > 0 else float('inf')
        epoch_bar.set_postfix({'loss': f'{avg_loss:.5f}', 'best': f'{best_loss:.5f}'})

        with open(loss_log_path, "a") as f:
            f.write(f"Epoch {epoch+1}\t{avg_loss:.6f}\n")

        
        if avg_loss < best_loss:
            best_loss = avg_loss
            
            # 1. Checkpoint
            # save_checkpoint(model, epoch + 1, avg_loss, ckpt_dir, saved_checkpoints, save_embedding=False)
            
            # 2. embedding  
            np.save(best_emb_path, model.rgcn.emb.weight.detach().cpu().numpy())

    
    print(f"\nTraining completed! Best loss: {best_loss:.6f}")
    print(f"Best Embeddings saved to: {best_emb_path}")
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate of pretrain')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='batch_size')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='number of epochs')
    ### graph classifier
    parser.add_argument('--in_size', type=int, default=256, 
                        help='The input feature size of graph learner')
    parser.add_argument('--hid_size', type=int, default=256, 
                        help='The input feature size of graph learner')
    parser.add_argument('--num_neg', type=int, default=16, 
                        help='The input feature size of graph learner')
    parser.add_argument('--model', type=str, default='SAGE', 
                        help='[HGAT, GCN, HGCN, GAT, LightGCN]')

    parser.add_argument('--score_file',type=str,default=None,
                        help='Path to the training scores file')
    args = parser.parse_args()
    

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print(args)
    print(f"Training with DGL built-in RGCN module")

    # load and preprocess dataset
    g = construct_dgl_graph(args)
    num_nodes = g.number_of_nodes()
    num_rels = 1
    print("Graph type:", type(g))
    
    
    sampler = NeighborSampler([10, 10]) 
    dataloader = DataLoader(
        g,
        torch.arange(g.number_of_nodes()),  
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0
    )

    
    pretrain_emb = load_pretrained_embeddings('ckpts/TransE_l2_Embedding_FP_512/Embedding_TransE_l2_entity.npy',
                                              device=device)
    
    print("Graph nodes:", g.number_of_nodes())
    print("Embedding shape:", pretrain_emb.shape)
    assert g.number_of_nodes() == pretrain_emb.shape[0], "Node count and embedding size mismatch!"

    model = LinkPredict(num_nodes, num_rels, h_dim=args.hid_size, pretrained_emb=pretrain_emb).to(device)

    train(
        dataloader,
        device,
        model,
        args
    )


# python RGCN.py --hid_size 512 --batch_size 3000 --score_file Data/Train/for_embedding/clean_reactions_scscore.txt 
