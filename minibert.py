import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MINI_BERT_CONFIG = {
    'vocab_size'              : 30522,   # Standard BERT vocab size
    'hidden_size'             : 512,     # 3x smaller than BERT-base (768)
    'num_layers'              : 6,       # 3x fewer layers than BERT-base (12)
    'num_heads'               : 8,       # 3x fewer heads than BERT-base (12)
    'intermediate_size'       : 1024,    # FFN expansion (2x hidden_size)
    'max_position_embeddings' : 512,     # Max sequence length for SEC/Earnings Tokenization
    'dropout'                 : 0.1,
}

# ─────────────────────────────────────────────
# 1. BERT EMBEDDINGS
# ─────────────────────────────────────────────
class BERTEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position, dropout):
        super().__init__()
        self.token    = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position = nn.Embedding(max_position, hidden_size)
        self.segment  = nn.Embedding(2, hidden_size)
        self.norm     = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, input_ids, segment_ids=None):
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)

        x = self.token(input_ids) + self.position(pos_ids) + self.segment(segment_ids)
        return self.dropout(self.norm(x))

# ─────────────────────────────────────────────
# 2. MULTI-HEAD SELF ATTENTION
# ─────────────────────────────────────────────
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout, use_xsa=False):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = hidden_size // num_heads
        self.use_xsa   = use_xsa   

        self.q        = nn.Linear(hidden_size, hidden_size)
        self.k        = nn.Linear(hidden_size, hidden_size)
        self.v        = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout  = nn.Dropout(dropout)

    def split_heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, attention_mask=None):
        B, T, _ = x.shape

        Q = self.split_heads(self.q(x))
        K = self.split_heads(self.k(x))
        V = self.split_heads(self.v(x))   

        scale  = math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        if attention_mask is not None:
            mask   = attention_mask[:, None, None, :]
            scores = scores.masked_fill(mask == 0, -1e4)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        Y = torch.matmul(attn_weights, V)   

        if self.use_xsa:
            V_self = V                                          
            V_norm = F.normalize(V_self, dim=-1)               
            proj   = (Y * V_norm).sum(dim=-1, keepdim=True)    
            Y      = Y - proj * V_norm                         

        context = Y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(context)

# ─────────────────────────────────────────────
# 3. FEED-FORWARD NETWORK
# ─────────────────────────────────────────────
class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout):
        super().__init__()
        self.fc1     = nn.Linear(hidden_size, intermediate_size)
        self.fc2     = nn.Linear(intermediate_size, hidden_size)
        self.act     = nn.GELU()   
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))

# ─────────────────────────────────────────────
# 4. TRANSFORMER BLOCK
# ─────────────────────────────────────────────
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout,use_xsa=False):
        super().__init__()
        self.attn    = MultiHeadSelfAttention(hidden_size, num_heads, dropout,use_xsa)
        self.ffn     = FeedForward(hidden_size, intermediate_size, dropout)
        self.norm1   = nn.LayerNorm(hidden_size, eps=1e-12)
        self.norm2   = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), attention_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

# ─────────────────────────────────────────────
# 5. CORE MINI BERT ARCHITECTURE
# ─────────────────────────────────────────────
class MiniBERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BERTEmbeddings(
            vocab_size   = config['vocab_size'],
            hidden_size  = config['hidden_size'],
            max_position = config['max_position_embeddings'],
            dropout      = config['dropout'],
        )
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size       = config['hidden_size'],
                num_heads         = config['num_heads'],
                intermediate_size = config['intermediate_size'],
                dropout           = config['dropout'],
                use_xsa           = config.get('use_xsa', False)
            )
            for _ in range(config['num_layers'])
        ])
        self.norm = nn.LayerNorm(config['hidden_size'], eps=1e-12)

    def forward(self, input_ids, attention_mask=None, segment_ids=None):
        x = self.embeddings(input_ids, segment_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.norm(x)
        cls_output = x[:, 0, :]   
        return x, cls_output

# ─────────────────────────────────────────────
# 6. MULTI-MODAL MINI BERT FOR PRETRAINING
#    (Phase 3 Loop: 2019-2025 Target)
# ─────────────────────────────────────────────
class MultiModalMiniBERTForPretraining(nn.Module):
    def __init__(self, config, num_tabular_features):
        super().__init__()
        self.bert = MiniBERT(config)

        # Tabular Processing Head (Outputs 128-dim)
        self.tabular_mlp = nn.Sequential(
            nn.Linear(num_tabular_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(256, 128),
            nn.LayerNorm(128)
        )

        # Objective A Head: MLM
        self.mlm_dense  = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.mlm_norm   = nn.LayerNorm(config['hidden_size'], eps=1e-12)
        self.mlm_act    = nn.GELU()
        self.mlm_head   = nn.Linear(config['hidden_size'], config['vocab_size'], bias=False)

        # Objective B Head: CTMM (Cross-Tabular-Modality-Matching)
        self.fused_dim = config['hidden_size'] + 128 
        self.ctmm_head = nn.Sequential(
            nn.Linear(self.fused_dim, self.fused_dim // 2),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(self.fused_dim // 2, 2) 
        )
        self.dropout = nn.Dropout(config['dropout'])

    def tie_weights(self):
        self.mlm_head.weight = self.bert.embeddings.token.weight

    def forward(self, input_ids, tabular_features, attention_mask=None, segment_ids=None, mlm_labels=None, ctmm_labels=None):
        sequence_output, cls_output = self.bert(input_ids, attention_mask, segment_ids)
        tabular_output = self.tabular_mlp(tabular_features)
        fused_tensor = torch.cat([cls_output, tabular_output], dim=-1)

        mlm_hidden = self.mlm_act(self.mlm_dense(sequence_output))
        mlm_hidden = self.mlm_norm(mlm_hidden)
        mlm_logits = self.mlm_head(self.dropout(mlm_hidden))

        ctmm_logits = self.ctmm_head(fused_tensor)

        total_loss = None
        mlm_loss   = None
        ctmm_loss  = None

        if mlm_labels is not None:
            mlm_loss = nn.CrossEntropyLoss()(
                mlm_logits.view(-1, mlm_logits.size(-1)), 
                mlm_labels.view(-1)
            )

        if ctmm_labels is not None:
            ctmm_loss = nn.CrossEntropyLoss()(ctmm_logits, ctmm_labels)

        if mlm_loss is not None and ctmm_loss is not None:
            total_loss = mlm_loss + ctmm_loss 
        elif mlm_loss is not None:
            total_loss = mlm_loss

        return {
            "loss"       : total_loss,
            "mlm_loss"   : mlm_loss,
            "ctmm_loss"  : ctmm_loss,
            "mlm_logits" : mlm_logits,
            "ctmm_logits": ctmm_logits,
        }

# ─────────────────────────────────────────────
# 7. MULTI-MODAL MINI BERT FOR CLASSIFICATION
#    (Phase 4 Loop: 2019-2025 Target)
# ─────────────────────────────────────────────
class MultiModalMiniBERTForClassification(nn.Module):
    def __init__(self, config, num_tabular_features, num_labels):
        super().__init__()
        self.bert = MiniBERT(config)

        self.tabular_mlp = nn.Sequential(
            nn.Linear(num_tabular_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(256, 128),
            nn.LayerNorm(128)
        )

        self.fused_dim = config['hidden_size'] + 128
        self.dropout = nn.Dropout(config['dropout'])
        self.classifier = nn.Linear(self.fused_dim, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, tabular_features, attention_mask=None, segment_ids=None, labels=None):
        _, cls_output = self.bert(input_ids, attention_mask, segment_ids)
        tabular_output = self.tabular_mlp(tabular_features)
        
        fused_tensor = torch.cat([cls_output, tabular_output], dim=-1)
        logits = self.classifier(self.dropout(fused_tensor))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {'loss': loss, 'logits': logits}

# ─────────────────────────────────────────────
# GPU SANITY CHECK
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("   MINIBERT MULTI-MODAL ARCHITECTURE COMPILATION TEST   ")
    print("=" * 70)

    # Auto-Detect NVIDIA GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" -> Execution Device: {device.type.upper()}")

    # Setup specific dimensions for our 500-company pipeline
    NUM_TABULAR_VARS = 5  # Volatility, Vol Trend, Momentum, Daily Range, Log Price
    NUM_CLASSES = 3       # DOWN(0), NEUTRAL(1), UP(2)
    
    model = MultiModalMiniBERTForClassification(
        MINI_BERT_CONFIG, 
        num_tabular_features=NUM_TABULAR_VARS, 
        num_labels=NUM_CLASSES
    ).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" -> Total parameters:     {total:,}")
    print(f" -> Trainable parameters: {trainable:,}")

    # Create dummy tensors mapped directly to the GPU
    batch_size, seq_len = 8, 128
    
    input_ids        = torch.randint(0, 30522, (batch_size, seq_len)).to(device)
    attention_mask   = torch.ones(batch_size, seq_len).long().to(device)
    tabular_features = torch.rand(batch_size, NUM_TABULAR_VARS).to(device) # The 5 Market vars
    labels           = torch.randint(0, NUM_CLASSES, (batch_size,)).to(device) # The 3 classes

    # Forward pass computation on GPU
    output = model(
        input_ids=input_ids, 
        tabular_features=tabular_features, 
        attention_mask=attention_mask, 
        labels=labels
    )
    
    print("\n--- FORWARD PASS RESULTS ---")
    print(f" -> Loss:         {output['loss'].item():.4f}")
    print(f" -> Logits shape: {output['logits'].shape} (Expected: {batch_size}, {NUM_CLASSES})") 
    print("\n ✅ GPU SANITY CHECK PASSED: Architecture is ready for training!")
    print("=" * 70)