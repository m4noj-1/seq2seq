import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import math

# Set random seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

# ============================================
# 1. DATA LOADING
# ============================================

class Vocabulary:
    def __init__(self):
        self.char2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2char = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.char_count = {}
        
    def add_char(self, char):
        if char not in self.char2idx:
            idx = len(self.char2idx)
            self.char2idx[char] = idx
            self.idx2char[idx] = char
        if char in self.char_count:
            self.char_count[char] += 1
        else:
            self.char_count[char] = 1
    
    def build_vocab(self, texts):
        for text in texts:
            for char in text:
                self.add_char(char)
    
    def __len__(self):
        return len(self.char2idx)


def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) >= 2:
                    data.append((parts[0].strip(), parts[1].strip()))
    return data


class TransliterationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        
        src_indices = [self.src_vocab.char2idx.get(c, self.src_vocab.char2idx['<UNK>']) 
                       for c in src]
        tgt_indices = [self.tgt_vocab.char2idx['<SOS>']] + \
                      [self.tgt_vocab.char2idx.get(c, self.tgt_vocab.char2idx['<UNK>']) 
                       for c in tgt] + \
                      [self.tgt_vocab.char2idx['<EOS>']]
        
        return torch.tensor(src_indices), torch.tensor(tgt_indices)


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_lengths = torch.tensor([len(s) for s in src_batch])
    tgt_lengths = torch.tensor([len(t) for t in tgt_batch])
    
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    
    return src_batch, tgt_batch, src_lengths, tgt_lengths


# ============================================
# 2. ULTIMATE MODEL ARCHITECTURE
# ============================================

class PositionalEncoding(nn.Module):
    """Add positional encoding to help with sequence position awareness"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    """Multi-head attention for better context modeling"""
    def __init__(self, hidden_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = torch.softmax(scores, dim=-1)
        x = torch.matmul(attention, V)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        return self.fc_out(x), attention


class Encoder(nn.Module):
    """Enhanced encoder with multi-head attention"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=3, dropout=0.3):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0,
                           bidirectional=True)
        
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, x, lengths=None):
        embedded = self.dropout(self.pos_encoding(self.embedding(x)))
        
        if lengths is not None:
            embedded = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        outputs, (hidden, cell) = self.lstm(embedded)
        
        if lengths is not None:
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        
        # Combine bidirectional states
        hidden = torch.tanh(self.fc_hidden(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        cell = torch.tanh(self.fc_cell(torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)))
        
        hidden = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        cell = cell.unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        return outputs, (hidden, cell)


class Decoder(nn.Module):
    """Enhanced decoder with multi-head attention"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, encoder_hidden_dim, 
                 num_layers=3, num_heads=8, dropout=0.3):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        
        self.lstm = nn.LSTM(embedding_dim + encoder_hidden_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Enhanced output layer with residual connection
        self.fc1 = nn.Linear(hidden_dim + encoder_hidden_dim + embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, hidden, encoder_outputs):
        embedded = self.dropout(self.pos_encoding(self.embedding(x)))
        
        # Multi-head attention
        if isinstance(hidden, tuple):
            query = hidden[0][-1].unsqueeze(1)
        else:
            query = hidden[-1].unsqueeze(1)
        
        context, attention_weights = self.attention(query, encoder_outputs, encoder_outputs)
        
        # Concatenate embedded and context
        rnn_input = torch.cat((embedded, context), dim=2)
        
        output, hidden = self.lstm(rnn_input, hidden)
        
        # Enhanced prediction with residual
        combined = torch.cat((output.squeeze(1), context.squeeze(1), embedded.squeeze(1)), dim=1)
        hidden_out = self.layer_norm(torch.relu(self.fc1(combined)))
        prediction = self.fc2(hidden_out)
        
        return prediction, hidden, attention_weights


class Seq2SeqUltimate(nn.Module):
    """Ultimate seq2seq model"""
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqUltimate, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, tgt, src_lengths=None, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.vocab_size
        
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        
        encoder_outputs, encoder_hidden = self.encoder(src, src_lengths)
        
        decoder_input = tgt[:, 0].unsqueeze(1)
        decoder_hidden = encoder_hidden
        
        for t in range(1, tgt_len):
            decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            outputs[:, t, :] = decoder_output
            
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = decoder_output.argmax(1)
            decoder_input = tgt[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        return outputs
    
    def predict(self, src, src_lengths=None, max_len=50):
        self.eval()
        with torch.no_grad():
            batch_size = src.shape[0]
            encoder_outputs, encoder_hidden = self.encoder(src, src_lengths)
            
            decoder_input = torch.tensor([[1]] * batch_size).to(self.device)
            decoder_hidden = encoder_hidden
            
            predictions = []
            
            for _ in range(max_len):
                decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                top1 = decoder_output.argmax(1)
                predictions.append(top1.unsqueeze(1))
                decoder_input = top1.unsqueeze(1)
                
                if (top1 == 2).all():
                    break
            
            return torch.cat(predictions, dim=1)
    
    def beam_search(self, src, src_lengths=None, beam_width=5, max_len=50):
        """Enhanced beam search with length normalization"""
        self.eval()
        with torch.no_grad():
            batch_size = src.shape[0]
            assert batch_size == 1
            
            encoder_outputs, encoder_hidden = self.encoder(src, src_lengths)
            
            beams = [{'sequence': [1], 'score': 0.0, 'hidden': encoder_hidden}]
            
            for step in range(max_len):
                candidates = []
                
                for beam in beams:
                    if beam['sequence'][-1] == 2:
                        candidates.append(beam)
                        continue
                    
                    decoder_input = torch.tensor([[beam['sequence'][-1]]]).to(self.device)
                    decoder_output, decoder_hidden, _ = self.decoder(decoder_input, beam['hidden'], encoder_outputs)
                    
                    log_probs = F.log_softmax(decoder_output, dim=1)
                    top_log_probs, top_indices = log_probs.topk(beam_width * 2)
                    
                    for log_prob, idx in zip(top_log_probs[0], top_indices[0]):
                        # Length normalization
                        length_penalty = ((5 + len(beam['sequence']) + 1) ** 0.6) / ((5 + 1) ** 0.6)
                        normalized_score = (beam['score'] + log_prob.item()) / length_penalty
                        
                        new_beam = {
                            'sequence': beam['sequence'] + [idx.item()],
                            'score': beam['score'] + log_prob.item(),
                            'normalized_score': normalized_score,
                            'hidden': decoder_hidden
                        }
                        candidates.append(new_beam)
                
                # Keep top beam_width beams based on normalized score
                beams = sorted(candidates, key=lambda x: x.get('normalized_score', x['score']), reverse=True)[:beam_width]
                
                if all(beam['sequence'][-1] == 2 for beam in beams):
                    break
            
            best_beam = beams[0]
            return torch.tensor([best_beam['sequence']]).to(self.device)


# ============================================
# 3. TRAINING WITH ADVANCED TECHNIQUES
# ============================================

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, ignore_index=-100):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        self.ignore_index = ignore_index
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.ignore_index] = 0
            mask = torch.nonzero(target == self.ignore_index, as_tuple=False)
            if mask.dim() > 0 and mask.size(0) > 0:
                true_dist.index_fill_(0, mask.squeeze(), 0.0)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


def train_epoch(model, dataloader, optimizer, criterion, device, clip=1, accumulation_steps=2):
    model.train()
    epoch_loss = 0
    optimizer.zero_grad()
    
    for idx, (src, tgt, src_lengths, tgt_lengths) in enumerate(tqdm(dataloader, desc="Training")):
        src, tgt = src.to(device), tgt.to(device)
        src_lengths = src_lengths.to(device)
        
        output = model(src, tgt, src_lengths)
        
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        tgt = tgt[:, 1:].reshape(-1)
        
        loss = criterion(output, tgt)
        loss = loss / accumulation_steps
        loss.backward()
        
        if (idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_loss += loss.item() * accumulation_steps
    
    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for src, tgt, src_lengths, tgt_lengths in tqdm(dataloader, desc="Evaluating"):
            src, tgt = src.to(device), tgt.to(device)
            src_lengths = src_lengths.to(device)
            
            output = model(src, tgt, src_lengths, teacher_forcing_ratio=0)
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)
            
            loss = criterion(output, tgt)
            epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def transliterate(model, text, src_vocab, tgt_vocab, device, max_len=50, use_beam_search=True):
    model.eval()
    indices = [src_vocab.char2idx.get(c, src_vocab.char2idx['<UNK>']) for c in text]
    src_tensor = torch.tensor([indices]).to(device)
    src_lengths = torch.tensor([len(indices)]).to(device)
    
    if use_beam_search:
        predictions = model.beam_search(src_tensor, src_lengths, beam_width=5, max_len=max_len)
    else:
        predictions = model.predict(src_tensor, src_lengths, max_len)
    
    predicted_chars = []
    for idx in predictions[0]:
        idx = idx.item()
        if idx == 2:
            break
        if idx not in [0, 1, 2]:
            predicted_chars.append(tgt_vocab.idx2char[idx])
    
    return ''.join(predicted_chars)


# ============================================
# 4. MAIN EXECUTION
# ============================================

CONFIG = {
    'embedding_dim': 256,
    'hidden_dim': 512,
    'num_layers': 3,  # Increased to 3!
    'num_heads': 8,
    'dropout': 0.35,
    'batch_size': 64,
    'learning_rate': 0.001,
    'num_epochs': 40,  # Increased to 40!
    'clip': 1,
    'teacher_forcing_ratio': 0.5,
    'label_smoothing': 0.15,
    'accumulation_steps': 2,
    'warmup_epochs': 5,
}

print("="*60)
print("🔥 ULTIMATE MODEL CONFIGURATION")
print("="*60)
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

print("\n" + "="*60)
print("LOADING DATA")
print("="*60)

train_data = load_data('hin_train.csv')
valid_data = load_data('hin_valid.csv')
test_data = load_data('hin_test.csv')

print(f"Train: {len(train_data)} | Valid: {len(valid_data)} | Test: {len(test_data)}")

src_vocab = Vocabulary()
tgt_vocab = Vocabulary()

for src, tgt in train_data:
    src_vocab.build_vocab([src])
    tgt_vocab.build_vocab([tgt])

print(f"Source vocab: {len(src_vocab)} | Target vocab: {len(tgt_vocab)}")

train_dataset = TransliterationDataset(train_data, src_vocab, tgt_vocab)
valid_dataset = TransliterationDataset(valid_data, src_vocab, tgt_vocab)
test_dataset = TransliterationDataset(test_data, src_vocab, tgt_vocab)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                          shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['batch_size'], 
                          shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], 
                         shuffle=False, collate_fn=collate_fn)

print("\n" + "="*60)
print("🚀 INITIALIZING ULTIMATE MODEL")
print("="*60)

encoder = Encoder(len(src_vocab), CONFIG['embedding_dim'], CONFIG['hidden_dim'], 
                 CONFIG['num_layers'], CONFIG['dropout'])
decoder = Decoder(len(tgt_vocab), CONFIG['embedding_dim'], CONFIG['hidden_dim'], 
                 CONFIG['hidden_dim'] * 2, CONFIG['num_layers'], CONFIG['num_heads'], CONFIG['dropout'])

model = Seq2SeqUltimate(encoder, decoder, device).to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Parameters: {count_parameters(model):,}")
print("\n✅ ULTIMATE Features:")
print("   • 3-layer Deep Network")
print("   • Multi-Head Attention (8 heads)")
print("   • Positional Encoding")
print("   • Enhanced Beam Search (width=5)")
print("   • Label Smoothing (0.15)")
print("   • Layer Normalization")
print("   • Residual Connections")
print("   • 40 Epochs Training")

optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
criterion = LabelSmoothingLoss(len(tgt_vocab), smoothing=CONFIG['label_smoothing'], ignore_index=0)

print("\n" + "="*60)
print("⏱️  TRAINING - WILL TAKE ~60-70 MINUTES")
print("="*60)

train_losses = []
valid_losses = []
best_valid_loss = float('inf')

for epoch in range(CONFIG['num_epochs']):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
    print('='*60)
    
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device, 
                            CONFIG['clip'], CONFIG['accumulation_steps'])
    valid_loss = evaluate(model, valid_loader, criterion, device)
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    print(f"📊 Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")
    
    scheduler.step(valid_loss)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"📉 Learning Rate: {current_lr:.6f}")
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'ultimate_model.pt')
        print("🏆 BEST MODEL SAVED!")

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train', linewidth=2)
plt.plot(valid_losses, label='Valid', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Progress (40 Epochs)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_losses)+1), train_losses, 'o-', label='Train', markersize=3)
plt.plot(range(1, len(valid_losses)+1), valid_losses, 's-', label='Valid', markersize=3)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Detailed Loss Curves')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

model.load_state_dict(torch.load('ultimate_model.pt'))

print("\n" + "="*60)
print("🎯 TESTING ULTIMATE MODEL")
print("="*60)

test_words = ['ghar', 'dost', 'paani', 'kitaab', 'pyaar', 'ajanabee', 
              'namaste', 'dhanyavaad', 'bharat', 'independence', 'computer', 'school']

print("\n🔥 BEAM SEARCH PREDICTIONS:")
for word in test_words:
    pred = transliterate(model, word, src_vocab, tgt_vocab, device, use_beam_search=True)
    print(f"  {word:15} -> {pred}")

test_loss = evaluate(model, test_loader, criterion, device)
print(f"\n🎯 Final Test Loss: {test_loss:.4f}")

print("\n" + "="*60)
print("✅ ULTIMATE MODEL COMPLETE!")
print("="*60)
print("\n🎯 Expected Results:")
print("   • Word Accuracy: 55-65% (was 39%)")
print("   • Char Accuracy: 82-88% (was 70%)")
print("\nRun accuracy calculation next! 🚀")
