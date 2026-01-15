'''
This is a standalone, importable SCScorer model. It does not have tensorflow as a
dependency and is a more attractive option for deployment. The calculations are
fast enough that there is no real reason to use GPUs (via tf) instead of CPUs (via np)
'''

import math, sys, random, os
import numpy as np
import time
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import json
import gzip
import six

import os
project_root = os.path.dirname(os.path.dirname(__file__))

score_scale = 5.0
min_separation = 0.25

FP_len = 1024
FP_rad = 2

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class SCScorer():
    def __init__(self, score_scale=score_scale):
        self.vars = []
        self.score_scale = score_scale
        self._restored = False

    def restore(self, weight_path=os.path.join(project_root, 'models', 'full_reaxys_model_1024bool', 'model.ckpt-10654.as_numpy.pickle'), FP_rad=FP_rad, FP_len=FP_len):
        self.FP_len = FP_len; self.FP_rad = FP_rad
        self._load_vars(weight_path)
        print('Restored variables from {}'.format(weight_path))

        if 'uint8' in weight_path or 'counts' in weight_path:
            def mol_to_fp(self, mol):
                if mol is None:
                    return np.array((self.FP_len,), dtype=np.uint8)
                fp = AllChem.GetMorganFingerprint(mol, self.FP_rad, useChirality=True) # uitnsparsevect
                fp_folded = np.zeros((self.FP_len,), dtype=np.uint8)
                for k, v in six.iteritems(fp.GetNonzeroElements()):
                    fp_folded[k % self.FP_len] += v
                return np.array(fp_folded)
        else:
            def mol_to_fp(self, mol):
                if mol is None:
                    return np.zeros((self.FP_len,), dtype=np.float32)
                return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, self.FP_rad, nBits=self.FP_len,
                    useChirality=True), dtype=bool)
        self.mol_to_fp = mol_to_fp

        self._restored = True
        return self

    def smi_to_fp(self, smi):
        if not smi:
            return np.zeros((self.FP_len,), dtype=np.float32)
        return self.mol_to_fp(self, Chem.MolFromSmiles(smi))

    def apply(self, x):
        if not self._restored:
            raise ValueError('Must restore model weights!')
        # Each pair of vars is a weight and bias term
        for i in range(0, len(self.vars), 2):
            last_layer = (i == len(self.vars)-2)
            W = self.vars[i]
            b = self.vars[i+1]
            x = np.matmul(x, W) + b
            if not last_layer:
                x = x * (x > 0) # ReLU
        x = 1 + (score_scale - 1) * sigmoid(x)
        return x

    def get_score_from_smi(self, smi='', v=False):
        if not smi:
            return ('', 0.)
        fp = np.array((self.smi_to_fp(smi)), dtype=np.float32)
        if sum(fp) == 0:
            if v: print('Could not get fingerprint?')
            cur_score = 0.
        else:
            # Run
            cur_score = self.apply(fp)
            if v: print('Score: {}'.format(cur_score))
        mol = Chem.MolFromSmiles(smi)
        if mol:
            smi = Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True)
        else:
            smi = ''
        return (smi, cur_score)

    def _load_vars(self, weight_path):
        if weight_path.endswith('pickle'):
            import  pickle
            with open(weight_path, 'rb') as fid:
                self.vars = pickle.load(fid)
                self.vars = [x.tolist() for x in self.vars]
        elif weight_path.endswith('json.gz'):
            with gzip.GzipFile(weight_path, 'r') as fin:    # 4. gzip
                json_bytes = fin.read()                      # 3. bytes (i.e. UTF-8)
                json_str = json_bytes.decode('utf-8')            # 2. string (i.e. JSON)
                self.vars = json.loads(json_str)
                self.vars = [np.array(x) for x in self.vars]


if __name__ == '__main__':
    # model = SCScorer()
    # model.restore(os.path.join(project_root, 'models', 'full_reaxys_model_1024bool', 'model.ckpt-10654.as_numpy.json.gz'))
    # smis = ['CCCOCCC', 'CCCNc1ccccc1']
    # for smi in smis:
    #     (smi, sco) = model.get_score_from_smi(smi)
    #     print('%.4f <--- %s' % (sco, smi))

    model = SCScorer()
    model.restore(os.path.join(project_root, 'USPTO-FULL', 'model.ckpt-10654.as_numpy.json.gz'), FP_len=2048)
    # c1ccc([O:5][C:3]([C@@:2]2([OH:1])[CH2:12][CH2:13][C@H:14]3[C@@H:15]4[CH2:16][CH2:17][C:18]5=[CH:19][C:20](=[O:30])[CH2:21][CH2:22][C@:23]5([CH3:24])[C:25]4=[CH:26][CH2:27][C@:28]23[CH3:29])=[CH2:4])cc1
    # .[OH-:45]
    # .C[C:48](=[O:49])[CH3:50]
    # >>[OH:1][C@:2]1([C:3](=[O:4])[CH2:5][O:49][C:48](=[O:45])[CH3:50])[CH2:12][CH2:13][C@H:14]2[C@@H:15]3[CH2:16][CH2:17][C:18]4=[CH:19][C:20](=[O:30])[CH2:21][CH2:22][C@:23]4([CH3:24])[C:25]3=[CH:26][CH2:27][C@:28]12[CH3:29]
    # smis = ['CCCOCCC', 'CCCNc1ccccc1']
    smis = ['[OH-:45]','C[C:48](=[O:49])[CH3:50]','c1ccc([O:5][C:3]([C@@:2]2([OH:1])[CH2:12][CH2:13][C@H:14]3[C@@H:15]4[CH2:16][CH2:17][C:18]5=[CH:19][C:20](=[O:30])[CH2:21][CH2:22][C@:23]5([CH3:24])[C:25]4=[CH:26][CH2:27][C@:28]23[CH3:29])=[CH2:4])cc1','[OH:1][C@:2]1([C:3](=[O:4])[CH2:5][O:49][C:48](=[O:45])[CH3:50])[CH2:12][CH2:13][C@H:14]2[C@@H:15]3[CH2:16][CH2:17][C:18]4=[CH:19][C:20](=[O:30])[CH2:21][CH2:22][C@:23]4([CH3:24])[C:25]3=[CH:26][CH2:27][C@:28]12[CH3:29]']
    for smi in smis:
        (smi, sco) = model.get_score_from_smi(smi)
        print('%.4f <--- %s' % (sco, smi))

    # model = SCScorer()
    # model.restore(os.path.join(project_root, 'models', 'full_reaxys_model_1024uint8', 'model.ckpt-10654.as_numpy.json.gz'))
    # smis = ['CCCOCCC', 'CCCNc1ccccc1']
    # for smi in smis:
    #     (smi, sco) = model.get_score_from_smi(smi)
    #     print('%.4f <--- %s' % (sco, smi))
