import os
import sys
import json
import re
import math
import random
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

from tqdm import tqdm


try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except Exception:
    SBERT_AVAILABLE = False

# Optional matplotlib dependency for loss curves
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Optional OpenAI client
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None
    OPENAI_AVAILABLE = False

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
if TORCH_AVAILABLE:
    torch.manual_seed(SEED)

# -------------------------
# Utilities
# -------------------------

def ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def clean_text(x) -> str:
    if x is None:
        return ''
    try:
        if pd.isna(x):
            return ''
    except Exception:
        pass
    s = str(x).strip()
    return '' if s.lower() == 'nan' else s

def get_first_existing(row: pd.Series, candidates: List[str], default: str = '') -> str:
    for c in candidates:
        if c in row.index:
            v = clean_text(row.get(c, ''))
            if v:
                return v
    return default

def normalize_basic_token(s: str) -> str:
    s = clean_text(s).lower().strip()
    aliases = {
        'js': 'javascript',
        'node': 'javascript',
        'nodejs': 'javascript',
        'ts': 'typescript',
        'py': 'python',
        'c++': 'cpp',
        'cplusplus': 'cpp',
        'c#': 'csharp',
        'c-sharp': 'csharp',
        'golang': 'go',
        'objective-c': 'objectivec',
        'obj-c': 'objectivec',
        'shell script': 'shell',
        'bash script': 'bash',
        'sh': 'shell',
    }
    return aliases.get(s, s)

def split_to_tokens(text: str) -> List[str]:
    s = clean_text(text)
    if not s:
        return []
    raw = re.split(r'[|,;/\
\t()]+', s)
    tokens = []
    for item in raw:
        item = normalize_basic_token(item)
        if item:
            tokens.append(item)
    return tokens

def extract_task_name(row: pd.Series) -> str:
    return get_first_existing(row, ['Task_name', 'name', 'task_name'])

def extract_task_language(row: pd.Series) -> str:
    return get_first_existing(row, ['programming_language', 'language'])

def extract_task_category(row: pd.Series) -> str:
    return get_first_existing(row, ['category'])

def extract_task_subcategory(row: pd.Series) -> str:
    return get_first_existing(row, ['subcategory', 'new_category'], default=extract_task_category(row))

def extract_task_theme(row: pd.Series) -> str:
    return get_first_existing(row, ['theme'])

def extract_mcp_category(row: pd.Series) -> str:
    return get_first_existing(row, ['category'])

def extract_mcp_subcategory(row: pd.Series) -> str:
    return get_first_existing(row, ['subcategory', 'new_category'], default=extract_mcp_category(row))

def extract_mcp_language(row: pd.Series) -> str:
    return get_first_existing(row, ['language'])

def extract_mcp_system(row: pd.Series) -> str:
    return get_first_existing(row, ['system'])

def task_metadata_dict(row: pd.Series) -> Dict[str, str]:
    return {
        'name': extract_task_name(row),
        'description': clean_text(row.get('description', '')),
        'language': extract_task_language(row),
        'category': extract_task_category(row),
        'subcategory': extract_task_subcategory(row),
        'theme': extract_task_theme(row),
    }

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim > 1:
        a = a.reshape(-1)
    if b.ndim > 1:
        b = b.reshape(-1)
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))

def save_loss_curve(losses: List[float], out_path: str, title: str = "Training Loss"):
    if not losses:
        return
    if not MATPLOTLIB_AVAILABLE:
        print('[WARN] matplotlib is unavailable.')
        return

    ensure_dir(os.path.dirname(out_path))
    epochs = list(range(1, len(losses) + 1))
    plt.figure()
    plt.plot(epochs, losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f'[PLOT] Saved loss curve to {out_path}')

# -------------------------
# Data loading and taxonomy structure
# -------------------------

def load_data(data_dir: str):
    mcp_raw = pd.read_csv(os.path.join(data_dir, 'mcp_raw.csv'))
    tasks = pd.read_csv(os.path.join(data_dir, 'task.csv'))
    mcp_task = pd.read_csv(os.path.join(data_dir, 'mcp_task.csv'))
    return mcp_raw, tasks, mcp_task

def build_category_tree(mcp_raw: pd.DataFrame, target_num_classes: int = 16) -> Dict:
    tree = {"root": {}}

    for idx, row in mcp_raw.iterrows():
        cat = extract_mcp_category(row) or 'Unknown'
        subcat = extract_mcp_subcategory(row) or cat or 'Unknown'
        if cat not in tree['root']:
            tree['root'][cat] = {}
        if subcat not in tree['root'][cat]:
            tree['root'][cat][subcat] = []
        tree['root'][cat][subcat].append(int(row.get('num', idx)))

    cat_cnt = len(tree['root'])
    subcat_cnt = sum(len(v) for v in tree['root'].values())
    return tree

def category_distance(cat_a: Tuple[str, str], cat_b: Tuple[str, str]) -> float:
    c1, s1 = cat_a
    c2, s2 = cat_b
    if c1 == c2:
        return 0.0 if s1 == s2 else 2.0
    return 4.0

# -------------------------
# Text and structural features
# -------------------------

def concat_mcp_text(row: pd.Series) -> str:
    parts = [
        clean_text(row.get('name', '')),
        clean_text(row.get('description', '')),
        extract_mcp_language(row),
        extract_mcp_system(row),
        clean_text(row.get('tools', '')),
        extract_mcp_category(row),
        extract_mcp_subcategory(row),
    ]
    return ' \n '.join([p for p in parts if p])

def concat_task_text(row: pd.Series) -> str:
    parts = [
        extract_task_name(row),
        clean_text(row.get('description', '')),
        extract_task_language(row),
        extract_task_category(row),
        extract_task_theme(row),
    ]
    return ' \n '.join([p for p in parts if p])

class TextEmbedder:
    def __init__(
        self,
        method: str = 'tfidf',
        sbert_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
        tfidf_max_features: int = 50000,
        tfidf_encode_batch_size: int = 512,
    ):
        self.method = method
        self._ready = False
        self.vectorizer = None
        self.model = None
        self.sbert_model_name = sbert_model
        self.tfidf_max_features = int(tfidf_max_features)
        self.tfidf_encode_batch_size = max(1, int(tfidf_encode_batch_size))

    def fit(self, texts: List[str]):
        if self.method == 'sbert' and SBERT_AVAILABLE:
            self.model = SentenceTransformer(self.sbert_model_name)
            self._ready = True
        else:
            self.vectorizer = TfidfVectorizer(max_features=self.tfidf_max_features, ngram_range=(1, 2))
            self.vectorizer.fit(texts)
            self._ready = True
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        assert self._ready, 'Call fit first.'
        if self.method == 'sbert' and SBERT_AVAILABLE:

            embs = self.model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        else:

            sparse_mat = self.vectorizer.transform(texts).astype(np.float32)
            out_chunks: List[np.ndarray] = []
            total = sparse_mat.shape[0]
            for start in tqdm(
                range(0, total, self.tfidf_encode_batch_size),
                total=(total + self.tfidf_encode_batch_size - 1) // self.tfidf_encode_batch_size,
                desc='[TFIDF] Dense encode',
                unit='batch',
                leave=False,
            ):
                end = min(start + self.tfidf_encode_batch_size, total)
                chunk = sparse_mat[start:end].toarray().astype(np.float32, copy=False)
                norms = np.linalg.norm(chunk, axis=1, keepdims=True) + 1e-12
                chunk = chunk / norms
                out_chunks.append(chunk)
            embs = np.vstack(out_chunks) if out_chunks else np.zeros((0, sparse_mat.shape[1]), dtype=np.float32)
        return embs

def language_compatible(task_language: str, mcp_language: str) -> float:
    task_tokens = {normalize_basic_token(x) for x in split_to_tokens(task_language)}
    mcp_tokens = {normalize_basic_token(x) for x in split_to_tokens(mcp_language)}
    if not task_tokens or not mcp_tokens:
        return 0.0
    return 1.0 if len(task_tokens & mcp_tokens) > 0 else 0.0

def theme_system_compatible(task_theme: str, mcp_system: str) -> float:
    theme = clean_text(task_theme).lower()
    system = clean_text(mcp_system).lower()
    if not theme or not system:
        return 0.0

    system_aliases = {
        'windows': ['windows', 'win'],
        'linux': ['linux', 'ubuntu', 'debian'],
        'ios': ['ios', 'iphone', 'ipad'],
        'macos': ['macos', 'mac', 'osx'],
        'android': ['android'],
        'docker': ['docker', 'container', 'kubernetes', 'k8s'],
        'web': ['web', 'browser', 'frontend', 'website'],
        'cloud': ['cloud', 'saas', 'serverless'],
    }
    matched_keys = []
    for key, aliases in system_aliases.items():
        if any(a in system for a in aliases):
            matched_keys.append(key)
    if not matched_keys:
        return 0.0

    theme_hits = 0
    for key in matched_keys:
        aliases = system_aliases[key]
        if any(a in theme for a in aliases):
            theme_hits += 1
    return 1.0 if theme_hits > 0 else 0.0

def structural_features(task_row: pd.Series, mcp_row: pd.Series) -> Dict[str, float]:
    task_cat = (extract_task_category(task_row) or 'Unknown', extract_task_subcategory(task_row) or 'Unknown')
    mcp_cat = (extract_mcp_category(mcp_row) or 'Unknown', extract_mcp_subcategory(mcp_row) or 'Unknown')

    d_cat = category_distance(mcp_cat, task_cat)
    phi_cat = 1.0 - (d_cat / 4.0)
    phi_lang = language_compatible(extract_task_language(task_row), extract_mcp_language(mcp_row))
    phi_theme = theme_system_compatible(extract_task_theme(task_row), extract_mcp_system(mcp_row))

    return {
        'd_cat': d_cat,
        'phi_cat': phi_cat,
        'phi_lang': phi_lang,
        'phi_theme': phi_theme,
    }

def structural_score(feats: Dict[str, float], w: Dict[str, float] = None) -> float:
    if w is None:
        w = {
            'phi_cat': 1.0 / 3.0,
            'phi_lang': 1.0 / 3.0,
            'phi_theme': 1.0 / 3.0,
        }
    weight_sum = sum(float(v) for v in w.values()) or 1.0
    return (
        w['phi_cat'] * feats['phi_cat'] +
        w['phi_lang'] * feats['phi_lang'] +
        w['phi_theme'] * feats['phi_theme']
    ) / weight_sum

# -------------------------
# Optional dual-tower model
# -------------------------

def build_mlp(dim_in: int, dim_hidden: int, dim_out: int, num_layers: int, dropout: float) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_dim = dim_in

    if num_layers <= 1:

        layers.append(nn.Linear(in_dim, dim_out))
    else:
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, dim_hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = dim_hidden
        layers.append(nn.Linear(in_dim, dim_out))

    return nn.Sequential(*layers)

class TwoTower(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int = 256,
        dim_out: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.task_tower = build_mlp(dim_in, dim_hidden, dim_out, num_layers, dropout)
        self.mcp_tower = build_mlp(dim_in, dim_hidden, dim_out, num_layers, dropout)

    def forward(self, t_vec: torch.Tensor, m_vec: torch.Tensor) -> torch.Tensor:
        t = self.task_tower(t_vec)
        m = self.mcp_tower(m_vec)

        t = nn.functional.normalize(t, dim=-1)
        m = nn.functional.normalize(m, dim=-1)
        return (t * m).sum(dim=-1)

def sample_pairs(mcp_task_df: pd.DataFrame, task_df: pd.DataFrame, mcp_df: pd.DataFrame, num_neg: int = 3) -> List[Tuple[int, int, int]]:
    # Positive mapping
    pos_map: Dict[int, set] = {}
    for _, row in mcp_task_df.iterrows():
        tid = int(row['task_id'])
        pos_map.setdefault(tid, set())
        for k in row.index:
            if k.startswith('mcp') and isinstance(row[k], (int, float)) and not math.isnan(row[k]):
                pos_map[tid].add(int(row[k]))

    all_mcps = list(set(mcp_df['num'].tolist()))
    out: List[Tuple[int, int, int]] = []
    for tid, pos_set in pos_map.items():
        for pm in pos_set:
            # Sample negatives
            neg_candidates = [m for m in all_mcps if m not in pos_set]
            random.shuffle(neg_candidates)
            for neg in neg_candidates[:num_neg]:
                out.append((tid, pm, neg))
    return out

def build_task_positive_map(mcp_task_df: pd.DataFrame) -> Dict[int, List[int]]:
    pos_map: Dict[int, List[int]] = {}
    for _, row in mcp_task_df.iterrows():
        tid = int(row['task_id'])
        pos_map.setdefault(tid, [])
        for c in row.index:
            if c.startswith('mcp') and not pd.isna(row[c]):
                pos_map[tid].append(int(row[c]))

    for tid in list(pos_map.keys()):
        seen = set()
        uniq = []
        for mid in pos_map[tid]:
            if mid not in seen:
                seen.add(mid)
                uniq.append(mid)
        pos_map[tid] = uniq
    return pos_map

def train_two_tower(
    task_vecs: Dict[int, np.ndarray],
    mcp_vecs: Dict[int, np.ndarray],
    pairs: List[Tuple[int, int, int]],
    epochs: int = 5,
    lr: float = 1e-3,
    gpu: int = 0,
    dim_hidden: int = 256,
    dim_out: int = 128,
    num_layers: int = 2,
    dropout: float = 0.1,
    batch_size: int = 512,
    output_dir: Optional[str] = None,
) -> Optional[TwoTower]:
    if not TORCH_AVAILABLE:
        print('[WARN] PyTorch is unavailable.')
        return None


    dim_in = next(iter(task_vecs.values())).shape[0]
    model = TwoTower(
        dim_in=dim_in,
        dim_hidden=dim_hidden,
        dim_out=dim_out,
        num_layers=num_layers,
        dropout=dropout,
    )

    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() and gpu >= 0 else 'cpu')
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    B = batch_size

    epoch_iter = tqdm(range(1, epochs + 1), desc='[TwoTower-BCE] Training', unit='epoch')

    epoch_losses: List[float] = []

    for ep in epoch_iter:
        random.shuffle(pairs)
        losses = []

        num_batches = (len(pairs) + B - 1) // B
        batch_iter = tqdm(
            range(0, len(pairs), B),
            desc=f'[BCE] Epoch {ep}/{epochs}',
            unit='batch',
            total=num_batches,
            leave=False
        )

        for i in batch_iter:
            batch = pairs[i:i+B]
            t_list, m_list, y_list = [], [], []
            for (tid, pm, nm) in batch:
                if tid not in task_vecs or pm not in mcp_vecs or nm not in mcp_vecs:
                    continue
                t_list.append(task_vecs[tid])
                m_list.append(mcp_vecs[pm])
                y_list.append(1.0)
                t_list.append(task_vecs[tid])
                m_list.append(mcp_vecs[nm])
                y_list.append(0.0)

            if not t_list:
                continue

            t = torch.tensor(np.stack(t_list), dtype=torch.float32, device=device)
            m = torch.tensor(np.stack(m_list), dtype=torch.float32, device=device)
            y = torch.tensor(y_list, dtype=torch.float32, device=device)

            opt.zero_grad()
            logits = model(t, m)
            loss = bce(logits, y)
            loss.backward()
            opt.step()

            loss_val = float(loss.detach().cpu().item())
            losses.append(loss_val)
            batch_iter.set_postfix({'loss': f'{loss_val:.4f}'})

        avg_loss = np.mean(losses) if losses else 0.0
        epoch_losses.append(avg_loss)
        epoch_iter.set_postfix({'avg_loss': f'{avg_loss:.4f}'})
        print(f"[TwoTower-BCE] epoch {ep}/{epochs} loss={avg_loss:.4f}")

    if output_dir is not None:
        loss_path = os.path.join(output_dir, 'two_tower_bce_loss.png')
        save_loss_curve(epoch_losses, loss_path, title='TwoTower-BCE Training Loss')

    return model

def train_two_tower_contrastive(
    task_vecs: Dict[int, np.ndarray],
    mcp_vecs: Dict[int, np.ndarray],
    task_pos_map: Dict[int, List[int]],
    epochs: int = 5,
    lr: float = 1e-3,
    gpu: int = 0,
    dim_hidden: int = 256,
    dim_out: int = 128,
    num_layers: int = 2,
    dropout: float = 0.1,
    batch_size: int = 256,
    temperature: float = 0.05,
    output_dir: Optional[str] = None,
) -> Optional[TwoTower]:
    if not TORCH_AVAILABLE:
        print('[WARN] PyTorch is unavailable.')
        return None

    dim_in = next(iter(task_vecs.values())).shape[0]
    model = TwoTower(
        dim_in=dim_in,
        dim_hidden=dim_hidden,
        dim_out=dim_out,
        num_layers=num_layers,
        dropout=dropout,
    )

    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() and gpu >= 0 else 'cpu')
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    B = batch_size

    valid_task_ids = [
        tid for tid, mids in task_pos_map.items()
        if tid in task_vecs and any(mid in mcp_vecs for mid in mids)
    ]
    if not valid_task_ids:
        print('[WARN] No valid tasks are available for contrastive learning.')
        return model

    epoch_iter = tqdm(range(1, epochs + 1), desc='[TwoTower-CL] Training', unit='epoch')
    epoch_losses: List[float] = []

    for ep in epoch_iter:
        random.shuffle(valid_task_ids)
        losses = []
        num_batches = (len(valid_task_ids) + B - 1) // B
        batch_iter = tqdm(
            range(0, len(valid_task_ids), B),
            desc=f'[CL] Epoch {ep}/{epochs}',
            unit='batch',
            total=num_batches,
            leave=False
        )

        for i in batch_iter:
            batch_task_ids = valid_task_ids[i:i+B]
            t_list, m_list = [], []
            for tid in batch_task_ids:
                pos_candidates = [mid for mid in task_pos_map.get(tid, []) if mid in mcp_vecs]
                if not pos_candidates:
                    continue
                mid = random.choice(pos_candidates)
                t_list.append(task_vecs[tid])
                m_list.append(mcp_vecs[mid])

            if len(t_list) < 2:
                continue

            t = torch.tensor(np.stack(t_list), dtype=torch.float32, device=device)
            m = torch.tensor(np.stack(m_list), dtype=torch.float32, device=device)

            t_emb = model.task_tower(t)
            m_emb = model.mcp_tower(m)
            t_emb = nn.functional.normalize(t_emb, dim=-1)
            m_emb = nn.functional.normalize(m_emb, dim=-1)

            logits = torch.matmul(t_emb, m_emb.t()) / temperature
            labels = torch.arange(logits.size(0), device=device)

            loss_1 = ce(logits, labels)
            loss_2 = ce(logits.t(), labels)
            loss = 0.5 * (loss_1 + loss_2)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_val = float(loss.detach().cpu().item())
            losses.append(loss_val)
            batch_iter.set_postfix({'cl_loss': f'{loss_val:.4f}'})

        avg_loss = float(np.mean(losses)) if losses else 0.0
        epoch_losses.append(avg_loss)
        epoch_iter.set_postfix({'avg_cl_loss': f'{avg_loss:.4f}'})
        print(f"[TwoTower-CL] epoch {ep}/{epochs} loss={avg_loss:.4f}")

    if output_dir is not None:
        loss_path = os.path.join(output_dir, 'two_tower_contrastive_loss.png')
        save_loss_curve(epoch_losses, loss_path, title='TwoTower-Contrastive Training Loss')

    return model

# -------------------------
# Ranking and candidate refinement
# -------------------------

def initial_ranking(task_row: pd.Series,
                    mcp_df: pd.DataFrame,
                    task_vec: np.ndarray,
                    mcp_vecs: Dict[int, np.ndarray],
                    alpha_semantic: float = 0.9,
                    alpha_struct: float = 0.1) -> List[Tuple[int, float]]:
    scores = []
    for _, mrow in mcp_df.iterrows():
        mid = int(mrow['num'])
        sem = cos_sim(task_vec, mcp_vecs[mid])
        feats = structural_features(task_row, mrow)
        stru = structural_score(feats)
        s = alpha_semantic * sem + alpha_struct * stru
        scores.append((mid, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def round2_expand(task_vec: np.ndarray,
                  picked: List[int],
                  mcp_vecs: Dict[int, np.ndarray],
                  topk_total: int = 20) -> List[int]:

    if not picked:
        return []
    picked_vecs = [mcp_vecs[mid] for mid in picked]
    centroid = np.mean(np.vstack([task_vec] + picked_vecs), axis=0)

    all_ids = list(mcp_vecs.keys())
    sims = []
    for mid in all_ids:
        if mid in picked:
            continue
        sims.append((mid, cos_sim(centroid, mcp_vecs[mid])))
    sims.sort(key=lambda x: x[1], reverse=True)
    need = max(0, topk_total - len(picked))
    return picked + [mid for (mid, _) in sims[:need]]

# -------------------------
# Evaluation metrics
# -------------------------

def dcg_at_k(rel: List[int], k: int) -> float:
    dcg = 0.0
    for i, r in enumerate(rel[:k]):
        if r:
            dcg += 1.0 / math.log2(i + 2)
    return dcg

def ndcg_at_k(rec_list: List[int], gt_set: set, k: int) -> float:
    rel = [1 if x in gt_set else 0 for x in rec_list[:k]]
    dcg = dcg_at_k(rel, k)
    ideal_hits = min(k, len(gt_set))
    idcg = dcg_at_k([1] * ideal_hits, k)
    return dcg / idcg if idcg > 0 else 0.0

def precision_recall_f1_at_k(rec_list: List[int], gt_set: set, k: int) -> Tuple[float, float, float]:
    rec_k = rec_list[:k]
    hits = sum(1 for x in rec_k if x in gt_set)
    precision = hits / max(1, k)
    recall = hits / max(1, len(gt_set))
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def hit_rate_at_k(rec_list: List[int], gt_set: set, k: int) -> float:
    rec_k = set(rec_list[:k])
    return 1.0 if len(rec_k & gt_set) > 0 else 0.0

# -------------------------
# Optional LLM re-ranking
# -------------------------
OPENAI_API_KEY_ENV = 'OPENAI_API_KEY'

def llm_self_check(
    task_text: str,
    mcp_table: pd.DataFrame,
    rec_ids: List[int],
    task_meta: Optional[Dict[str, str]] = None,
) -> List[int]:
    api_key = os.environ.get(OPENAI_API_KEY_ENV)
    if not OPENAI_AVAILABLE or not api_key or not rec_ids:
        return rec_ids

    try:
        client_kwargs = {'api_key': api_key}
        base_url = os.environ.get('OPENAI_BASE_URL', '').strip()
        if base_url:
            client_kwargs['base_url'] = base_url
        client = OpenAI(**client_kwargs)

        candidate_cards = []
        for mid in rec_ids:
            sub = mcp_table.loc[mcp_table['num'] == mid]
            if sub.empty:
                continue
            row = sub.iloc[0]
            candidate_cards.append({
                'id': int(mid),
                'name': clean_text(row.get('name', '')),
                'description': clean_text(row.get('description', '')),
                'language': extract_mcp_language(row),
                'system': extract_mcp_system(row),
                'category': extract_mcp_category(row),
                'subcategory': extract_mcp_subcategory(row),
                'tools': clean_text(row.get('tools', '')),
                'official': clean_text(row.get('official', '')),
                'license': clean_text(row.get('license', '')),
            })

        if len(candidate_cards) != len(rec_ids):
            return rec_ids

        allowed_ids = [card['id'] for card in candidate_cards]
        allowed_set = set(allowed_ids)

        prompt_payload = {
            'task': {
                'text': task_text,
                'meta': task_meta or {},
                'K': len(rec_ids),
            },
            'ranking_criteria': [
                'Closely match the task intent and required capabilities.',
                'Satisfy engineering constraints such as language and environment when supported by the candidate metadata.',
                'Fit naturally into the intended execution workflow.',
                'Prefer complementary but non-redundant candidates.',
                'Avoid weakly relevant or overly generic candidates.',
            ],
            'rules': {
                'return_valid_json_only': True,
                'return_exactly_k_ids': len(rec_ids),
                'must_use_only_candidate_ids_from_pool': True,
                'must_not_add_or_remove_ids': True,
                'preserve_identifier_values_exactly': True,
                'output_schema': {
                    'MCP_servers': ['id1', 'id2', '...'],
                    'Explanation': 'One short sentence.'
                }
            },
            'candidate_cards': candidate_cards,
        }

        model_candidates = []
        for item in [os.environ.get('OPENAI_MODEL', '').strip(), 'gpt-5.2', 'gpt-5.4']:
            if item and item not in model_candidates:
                model_candidates.append(item)

        text_out = None
        last_error = None
        for model_name in model_candidates:
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            'role': 'system',
                            'content': (
                                'You are a constrained list-wise reranker for task-oriented MCP server recommendation. '
                                'Return strict JSON only with keys MCP_servers and Explanation.'
                            )
                        },
                        {
                            'role': 'user',
                            'content': json.dumps(prompt_payload, ensure_ascii=False)
                        }
                    ],
                    temperature=0,
                )
                text_out = resp.choices[0].message.content
                if text_out:
                    break
            except Exception as e:
                last_error = e
                continue

        if not text_out:
            raise RuntimeError(f'No valid LLM response. last_error={last_error}')

        parsed = None
        try:
            parsed = json.loads(text_out)
        except Exception:
            match = re.search(r'\{.*\}', text_out, flags=re.S)
            if match:
                parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            return rec_ids

        ids = parsed.get('MCP_servers', [])
        if not isinstance(ids, list):
            return rec_ids

        final_ids = []
        seen = set()
        for x in ids:
            try:
                mid = int(x)
            except Exception:
                continue
            if mid in allowed_set and mid not in seen:
                seen.add(mid)
                final_ids.append(mid)

        if len(final_ids) != len(rec_ids):
            return rec_ids
        if set(final_ids) != set(allowed_ids):
            return rec_ids
        return final_ids
    except Exception as e:
        print(f"[LLM] Re-ranking failed; falling back to the original order: {e}")
        return rec_ids

# -------------------------
# Split helpers and embeddings
# -------------------------
def split_by_task(tasks: pd.DataFrame, train_ratio=0.6, val_ratio=0.2):
    ids = tasks['task_id'].tolist() if 'task_id' in tasks.columns else tasks['Task_id'].tolist()
    train_ids, temp_ids = train_test_split(ids, test_size=1-train_ratio, random_state=SEED)
    val_size = val_ratio / (1 - train_ratio)
    val_ids, test_ids = train_test_split(temp_ids, test_size=1-val_size, random_state=SEED)
    return set(train_ids), set(val_ids), set(test_ids)

def filter_interactions_by_task_ids(mcp_task_df: pd.DataFrame, task_ids: set) -> pd.DataFrame:
    return mcp_task_df[mcp_task_df['task_id'].isin(task_ids)].reset_index(drop=True)

def build_embeddings(
    mcp_df: pd.DataFrame,
    task_df_all: pd.DataFrame,
    method: str = 'tfidf',
    sbert_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
    fit_task_df: Optional[pd.DataFrame] = None,
    fit_mcp_df: Optional[pd.DataFrame] = None,
    tfidf_max_features: int = 50000,
    tfidf_encode_batch_size: int = 512,
):
    mcp_df = mcp_df.copy()
    task_df_all = task_df_all.copy()
    fit_task_df = task_df_all.copy() if fit_task_df is None else fit_task_df.copy()
    fit_mcp_df = mcp_df.copy() if fit_mcp_df is None else fit_mcp_df.copy()

    mcp_df['__text__'] = mcp_df.apply(concat_mcp_text, axis=1)
    task_df_all['__text__'] = task_df_all.apply(concat_task_text, axis=1)
    fit_task_df['__text__'] = fit_task_df.apply(concat_task_text, axis=1)
    fit_mcp_df['__text__'] = fit_mcp_df.apply(concat_mcp_text, axis=1)

    embedder = TextEmbedder(
        method=method,
        sbert_model=sbert_model,
        tfidf_max_features=tfidf_max_features,
        tfidf_encode_batch_size=tfidf_encode_batch_size,
    ).fit(
        fit_mcp_df['__text__'].tolist() + fit_task_df['__text__'].tolist()
    )
    mcp_emb = embedder.encode(mcp_df['__text__'].tolist())
    task_emb = embedder.encode(task_df_all['__text__'].tolist())

    mcp_ids = mcp_df['num'].tolist()
    task_id_col = 'task_id' if 'task_id' in task_df_all.columns else 'Task_id'
    task_ids = task_df_all[task_id_col].tolist()

    mcp_vecs = {int(i): v for i, v in zip(mcp_ids, mcp_emb)}
    task_vecs = {int(i): v for i, v in zip(task_ids, task_emb)}
    return embedder, mcp_vecs, task_vecs

# -------------------------
# Split-level evaluation
# -------------------------

def eval_on_split(
    tasks_df: pd.DataFrame,
    mcp_df: pd.DataFrame,
    mcp_task_df: pd.DataFrame,
    task_vecs: Dict[int, np.ndarray],
    mcp_vecs: Dict[int, np.ndarray],
    topk1: int = 5,
    topk2: int = 20,
    alpha_semantic: float = 0.9,
    alpha_struct: float = 0.1,
    use_llm_selfcheck: bool = False,
    split_name: str = "SPLIT",
) -> Tuple[Dict[str, float], pd.DataFrame]:
    gt: Dict[int, set] = {}
    for _, row in mcp_task_df.iterrows():
        tid = int(row['task_id'])
        pos = set()
        for c in row.index:
            if c.startswith('mcp') and not pd.isna(row[c]):
                pos.add(int(row[c]))
        if pos:
            gt[tid] = pos


    metric_keys = [
        'NDCG@5', 'NDCG@10',
        'Recall@5', 'Recall@10',
        'Precision@5', 'Precision@10',
        'F1@5', 'F1@10'
    ]
    metrics = {k: 0.0 for k in metric_keys}
    n = 0

    rec_rows = []


    task_iter = tqdm(
        tasks_df.iterrows(),
        total=len(tasks_df),
        desc=f"[{split_name}] Epoch 1/1",
        unit="task"
    )

    for _, trow in task_iter:
        task_id_col = 'task_id' if 'task_id' in trow.index else 'Task_id'
        tid = int(trow[task_id_col])
        if tid not in task_vecs:
            continue
        tvec = task_vecs[tid]
        # Initial ranking
        init_scores = initial_ranking(trow, mcp_df, tvec, mcp_vecs, alpha_semantic, alpha_struct)
        top_list_k1 = [mid for (mid, _) in init_scores[:topk1]]
        # Second-stage expansion
        rec_list = round2_expand(tvec, top_list_k1, mcp_vecs, topk_total=topk2)
        # Optional LLM re-ranking
        if use_llm_selfcheck:
            rec_list = llm_self_check(concat_task_text(trow), mcp_df, rec_list, task_meta=task_metadata_dict(trow))

        # Evaluate
        gt_set = gt.get(tid, set())

        for k in [5, 10]:
            ndcg = ndcg_at_k(rec_list, gt_set, k)
            p, r, f = precision_recall_f1_at_k(rec_list, gt_set, k)

            metrics[f'NDCG@{k}'] += ndcg
            metrics[f'Recall@{k}'] += r
            metrics[f'Precision@{k}'] += p
            metrics[f'F1@{k}'] += f

        n += 1

        rec_rows.append({
            'task_id': tid,
            'top5': json.dumps(top_list_k1, ensure_ascii=False),
            'top10': json.dumps(rec_list[:10], ensure_ascii=False)
        })

    if n > 0:
        for k in metrics:
            metrics[k] /= n
    else:
        print(f'[WARN] [{split_name}] No evaluable samples were found.')

    return metrics, pd.DataFrame(rec_rows)

# -------------------------
# Main entry point
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--embedding_type', type=str, default='tfidf', choices=['tfidf', 'sbert'])
    parser.add_argument('--sbert_model', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--fit_all_mcp_for_embedding', type=int, default=1, help='1=fit representation with all known MCP texts')
    parser.add_argument('--tfidf_max_features', type=int, default=50000, help='max TF-IDF features')
    parser.add_argument('--tfidf_encode_batch_size', type=int, default=512, help='batch size for TF-IDF sparse->dense encoding')
    parser.add_argument('--alpha_semantic', type=float, default=0.9)
    parser.add_argument('--alpha_struct', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_llm_selfcheck', type=int, default=0)
    parser.add_argument('--topk1', type=int, default=5)
    parser.add_argument('--topk2', type=int, default=20)
    parser.add_argument('--use_two_tower', type=int, default=0)

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for two-tower training')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden dimension of MLP in two-tower')
    parser.add_argument('--out_dim', type=int, default=256, help='output dimension of two-tower embeddings')
    parser.add_argument('--num_layers', type=int, default=3, help='number of layers in each tower MLP')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate in two-tower MLP')

    parser.add_argument('--loss_type', type=str, default='contrastive',
                        choices=['bce', 'contrastive'],
                        help='two-tower loss: bce or contrastive')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='temperature for contrastive loss')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size for two-tower training')
    parser.add_argument('--num_neg', type=int, default=10,
                        help='number of sampled negatives per positive for BCE loss')

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    script_name = os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else 'interactive'
    print(f"[CONFIG] script={script_name} args={json.dumps(vars(args), ensure_ascii=False)}")
    if args.use_llm_selfcheck:
        print('[NOTE] Constrained LLM re-ranking is enabled.')

    mcp_raw, tasks, mcp_task = load_data(args.data_dir)
    _ = build_category_tree(mcp_raw, target_num_classes=16)

    train_ids, val_ids, test_ids = split_by_task(tasks, 0.6, 0.2)
    task_id_col = 'task_id' if 'task_id' in tasks.columns else 'Task_id'
    tasks_train = tasks[tasks[task_id_col].isin(train_ids)].reset_index(drop=True)
    tasks_val = tasks[tasks[task_id_col].isin(val_ids)].reset_index(drop=True)
    tasks_test = tasks[tasks[task_id_col].isin(test_ids)].reset_index(drop=True)

    mcp_task_train = filter_interactions_by_task_ids(mcp_task, train_ids)
    mcp_task_val = filter_interactions_by_task_ids(mcp_task, val_ids)
    mcp_task_test = filter_interactions_by_task_ids(mcp_task, test_ids)

    fit_mcp_df = mcp_raw if bool(args.fit_all_mcp_for_embedding) else mcp_raw
    embedder, mcp_vecs, task_vecs = build_embeddings(
        mcp_raw,
        tasks,
        method=args.embedding_type,
        sbert_model=args.sbert_model,
        fit_task_df=tasks_train,
        fit_mcp_df=fit_mcp_df,
        tfidf_max_features=args.tfidf_max_features,
        tfidf_encode_batch_size=args.tfidf_encode_batch_size,
    )

    if args.use_two_tower:
        if args.loss_type == 'contrastive':
            task_pos_map = build_task_positive_map(mcp_task_train)
            model = train_two_tower_contrastive(
                task_vecs, mcp_vecs, task_pos_map,
                epochs=args.epochs,
                lr=args.lr,
                gpu=args.gpu,
                dim_hidden=args.hidden_dim,
                dim_out=args.out_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                batch_size=args.batch_size,
                temperature=args.temperature,
                output_dir=args.output_dir,
            )
        else:
            pairs = sample_pairs(mcp_task_train, tasks_train, mcp_raw, num_neg=args.num_neg)
            model = train_two_tower(
                task_vecs, mcp_vecs, pairs,
                epochs=args.epochs,
                lr=args.lr,
                gpu=args.gpu,
                dim_hidden=args.hidden_dim,
                dim_out=args.out_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                batch_size=args.batch_size,
                output_dir=args.output_dir,
            )

        if model is not None and TORCH_AVAILABLE:
            device = next(model.parameters()).device
            model.eval()

            def project(vecs: Dict[int, np.ndarray], tower: nn.Sequential) -> Dict[int, np.ndarray]:
                out: Dict[int, np.ndarray] = {}
                with torch.no_grad():
                    for k, v in vecs.items():
                        x = torch.tensor(v[None, :], dtype=torch.float32, device=device)
                        y = tower(x)
                        y = nn.functional.normalize(y, dim=-1)
                        out[k] = y.detach().cpu().numpy().reshape(-1)
                return out

            task_vecs = project(task_vecs, model.task_tower)
            mcp_vecs = project(mcp_vecs, model.mcp_tower)

    train_metrics, _ = eval_on_split(
        tasks_train, mcp_raw, mcp_task_train, task_vecs, mcp_vecs,
        topk1=args.topk1, topk2=args.topk2,
        alpha_semantic=args.alpha_semantic, alpha_struct=args.alpha_struct,
        use_llm_selfcheck=bool(args.use_llm_selfcheck),
        split_name="TRAIN"
    )
    print('[TRAIN]', json.dumps(train_metrics, ensure_ascii=False, indent=2))

    val_metrics, _ = eval_on_split(
        tasks_val, mcp_raw, mcp_task_val, task_vecs, mcp_vecs,
        topk1=args.topk1, topk2=args.topk2,
        alpha_semantic=args.alpha_semantic, alpha_struct=args.alpha_struct,
        use_llm_selfcheck=bool(args.use_llm_selfcheck),
        split_name="VAL"
    )
    print('[VAL]', json.dumps(val_metrics, ensure_ascii=False, indent=2))

    test_metrics, recs_test = eval_on_split(
        tasks_test, mcp_raw, mcp_task_test, task_vecs, mcp_vecs,
        topk1=args.topk1, topk2=args.topk2,
        alpha_semantic=args.alpha_semantic, alpha_struct=args.alpha_struct,
        use_llm_selfcheck=bool(args.use_llm_selfcheck),
        split_name="TEST"
    )
    print('[TEST]', json.dumps(test_metrics, ensure_ascii=False, indent=2))

    out_path = os.path.join(args.output_dir, 'recs_test.csv')
    recs_test.to_csv(out_path, index=False)
    print(f'[SAVE] {out_path}')

    summary = {
        "script": script_name,
        "args": vars(args),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

if __name__ == '__main__':
    main()
