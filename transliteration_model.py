import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================

class Vocabulary:
    """Build vocabulary from the dataset"""
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
    """Load data from CSV file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(',')
                if len(parts) >= 2:
                    # Latin script, Devanagari script
                    data.append((parts[0].strip(), parts[1].strip()))
    return data


class TransliterationDataset(Dataset):
    """Custom Dataset for transliteration"""
    def __init__(self, data, src_vocab, tgt_vocab):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        
        # Convert to indices
        src_indices = [self.src_vocab.char2idx.get(c, self.src_vocab.char2idx['<UNK>']) 
                       for c in src]
        tgt_indices = [self.tgt_vocab.char2idx['<SOS>']] + \
                      [self.tgt_vocab.char2idx.get(c, self.tgt_vocab.char2idx['<UNK>']) 
                       for c in tgt] + \
                      [self.tgt_vocab.char2idx['<EOS>']]
        
        return torch.tensor(src_indices), torch.tensor(tgt_indices)


def collate_fn(batch):
    """Collate function to pad sequences in a batch"""
    src_batch, tgt_batch = zip(*batch)
    
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    
    return src_batch, tgt_batch


# ============================================
# 2. MODEL ARCHITECTURE
# ============================================

class Encoder(nn.Module):
    """Encoder RNN"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, 
                 cell_type='LSTM', dropout=0.1):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        if cell_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, 
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:  # RNN
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, 
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        if self.cell_type == 'LSTM':
            outputs, (hidden, cell) = self.rnn(embedded)
            return outputs, (hidden, cell)
        else:
            outputs, hidden = self.rnn(embedded)
            return outputs, hidden


class Decoder(nn.Module):
    """Decoder RNN"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, 
                 cell_type='LSTM', dropout=0.1):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        if cell_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, 
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:  # RNN
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, 
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden):
        # x: (batch_size, 1)
        embedded = self.embedding(x)  # (batch_size, 1, embedding_dim)
        
        if self.cell_type == 'LSTM':
            output, (hidden, cell) = self.rnn(embedded, hidden)
            prediction = self.fc(output.squeeze(1))  # (batch_size, vocab_size)
            return prediction, (hidden, cell)
        else:
            output, hidden = self.rnn(embedded, hidden)
            prediction = self.fc(output.squeeze(1))  # (batch_size, vocab_size)
            return prediction, hidden


class Seq2Seq(nn.Module):
    """Sequence to Sequence Model"""
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # src: (batch_size, src_len)
        # tgt: (batch_size, tgt_len)
        
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.fc.out_features
        
        # Store outputs
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        
        # Encode the source sequence
        _, encoder_hidden = self.encoder(src)
        
        # First input to decoder is <SOS> token
        decoder_input = tgt[:, 0].unsqueeze(1)  # (batch_size, 1)
        decoder_hidden = encoder_hidden
        
        # Decode one character at a time
        for t in range(1, tgt_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[:, t, :] = decoder_output
            
            # Teacher forcing: use actual target as next input
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = decoder_output.argmax(1)
            decoder_input = tgt[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        return outputs
    
    def predict(self, src, max_len=50):
        """Predict without teacher forcing"""
        self.eval()
        with torch.no_grad():
            batch_size = src.shape[0]
            
            # Encode
            _, encoder_hidden = self.encoder(src)
            
            # Start with <SOS>
            decoder_input = torch.tensor([[1]] * batch_size).to(self.device)  # <SOS>
            decoder_hidden = encoder_hidden
            
            predictions = []
            
            for _ in range(max_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                top1 = decoder_output.argmax(1)
                predictions.append(top1.unsqueeze(1))
                
                decoder_input = top1.unsqueeze(1)
                
                # Stop if all sequences predict <EOS>
                if (top1 == 2).all():  # <EOS> token
                    break
            
            return torch.cat(predictions, dim=1)


# ============================================
# 3. TRAINING FUNCTIONS
# ============================================

def train_epoch(model, dataloader, optimizer, criterion, device, clip=1):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    
    for src, tgt in tqdm(dataloader, desc="Training"):
        src, tgt = src.to(device), tgt.to(device)
        
        optimizer.zero_grad()
        
        output = model(src, tgt)
        
        # Reshape for loss calculation
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        tgt = tgt[:, 1:].reshape(-1)
        
        loss = criterion(output, tgt)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Evaluating"):
            src, tgt = src.to(device), tgt.to(device)
            
            output = model(src, tgt, teacher_forcing_ratio=0)  # No teacher forcing
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)
            
            loss = criterion(output, tgt)
            epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def transliterate(model, text, src_vocab, tgt_vocab, device, max_len=50):
    """Transliterate a single text"""
    model.eval()
    
    # Convert text to indices
    indices = [src_vocab.char2idx.get(c, src_vocab.char2idx['<UNK>']) for c in text]
    src_tensor = torch.tensor([indices]).to(device)
    
    # Get prediction
    predictions = model.predict(src_tensor, max_len)
    
    # Convert indices back to characters
    predicted_chars = []
    for idx in predictions[0]:
        idx = idx.item()
        if idx == 2:  # <EOS>
            break
        if idx not in [0, 1, 2]:  # Skip <PAD>, <SOS>, <EOS>
            predicted_chars.append(tgt_vocab.idx2char[idx])
    
    return ''.join(predicted_chars)


# ============================================
# 4. MAIN EXECUTION
# ============================================

# CONFIGURATION - You can change these parameters
CONFIG = {
    'embedding_dim': 128,
    'hidden_dim': 256,
    'num_layers': 1,
    'cell_type': 'LSTM',  # Options: 'LSTM', 'GRU', 'RNN'
    'dropout': 0.1,
    'batch_size': 64,
    'learning_rate': 0.001,
    'num_epochs': 10,
    'clip': 1,
}

print("Configuration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# Load data - UPLOAD YOUR FILES TO COLAB FIRST
print("\n" + "="*50)
print("LOADING DATA")
print("="*50)

# Upload files to Colab (you'll do this manually in Colab)
# For now, assuming files are in the current directory
train_data = load_data('hin_train.csv')
valid_data = load_data('hin_valid.csv')
test_data = load_data('hin_test.csv')

print(f"Train samples: {len(train_data)}")
print(f"Valid samples: {len(valid_data)}")
print(f"Test samples: {len(test_data)}")
print(f"\nExample: {train_data[0][0]} -> {train_data[0][1]}")

# Build vocabularies
print("\nBuilding vocabularies...")
src_vocab = Vocabulary()
tgt_vocab = Vocabulary()

for src, tgt in train_data:
    src_vocab.build_vocab([src])
    tgt_vocab.build_vocab([tgt])

print(f"Source vocab size: {len(src_vocab)}")
print(f"Target vocab size: {len(tgt_vocab)}")

# Create datasets
train_dataset = TransliterationDataset(train_data, src_vocab, tgt_vocab)
valid_dataset = TransliterationDataset(valid_data, src_vocab, tgt_vocab)
test_dataset = TransliterationDataset(test_data, src_vocab, tgt_vocab)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                          shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['batch_size'], 
                          shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], 
                         shuffle=False, collate_fn=collate_fn)

# Initialize model
print("\n" + "="*50)
print("INITIALIZING MODEL")
print("="*50)

encoder = Encoder(len(src_vocab), CONFIG['embedding_dim'], CONFIG['hidden_dim'], 
                 CONFIG['num_layers'], CONFIG['cell_type'], CONFIG['dropout'])
decoder = Decoder(len(tgt_vocab), CONFIG['embedding_dim'], CONFIG['hidden_dim'], 
                 CONFIG['num_layers'], CONFIG['cell_type'], CONFIG['dropout'])

model = Seq2Seq(encoder, decoder, device).to(device)

# Count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total trainable parameters: {count_parameters(model):,}")

# Initialize optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding

# Training loop
print("\n" + "="*50)
print("TRAINING")
print("="*50)

train_losses = []
valid_losses = []
best_valid_loss = float('inf')

for epoch in range(CONFIG['num_epochs']):
    print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
    
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device, CONFIG['clip'])
    valid_loss = evaluate(model, valid_loader, criterion, device)
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    
    print(f"Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")
    
    # Save best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'best_model.pt')
        print("  → Best model saved!")

# Plot losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.grid(True)
plt.show()

# Load best model for testing
model.load_state_dict(torch.load('best_model.pt'))

# Test the model
print("\n" + "="*50)
print("TESTING")
print("="*50)

test_examples = ['ghar', 'dost', 'paani', 'kitaab', 'pyaar', 'ajanabee']

print("\nTransliteration Examples:")
for word in test_examples:
    prediction = transliterate(model, word, src_vocab, tgt_vocab, device)
    print(f"  {word} -> {prediction}")

# Calculate test loss
test_loss = evaluate(model, test_loader, criterion, device)
print(f"\nTest Loss: {test_loss:.4f}")

print("\n" + "="*50)
print("ANALYSIS: COMPUTATIONS AND PARAMETERS")
print("="*50)

# For the theoretical analysis (assuming single layer, same vocab size)
print("\nAssuming:")
print(f"  - Embedding dimension (E) = {CONFIG['embedding_dim']}")
print(f"  - Hidden dimension (H) = {CONFIG['hidden_dim']}")
print(f"  - Number of layers = 1")
print(f"  - Sequence length (L) = variable")
print(f"  - Vocab size source (V_src) = {len(src_vocab)}")
print(f"  - Vocab size target (V_tgt) = {len(tgt_vocab)}")

E = CONFIG['embedding_dim']
H = CONFIG['hidden_dim']
V_src = len(src_vocab)
V_tgt = len(tgt_vocab)

print("\n1. TOTAL PARAMETERS:")
print("   Components:")
print(f"   - Encoder Embedding: V_src × E = {V_src} × {E} = {V_src * E:,}")
print(f"   - Encoder LSTM: 4 × (E×H + H×H + H) = 4 × ({E}×{H} + {H}×{H} + {H}) = {4 * (E*H + H*H + H):,}")
print(f"   - Decoder Embedding: V_tgt × E = {V_tgt} × {E} = {V_tgt * E:,}")
print(f"   - Decoder LSTM: 4 × (E×H + H×H + H) = {4 * (E*H + H*H + H):,}")
print(f"   - Decoder FC: H × V_tgt + V_tgt = {H} × {V_tgt} + {V_tgt} = {H * V_tgt + V_tgt:,}")
total_params = V_src*E + 4*(E*H + H*H + H) + V_tgt*E + 4*(E*H + H*H + H) + H*V_tgt + V_tgt
print(f"   TOTAL: {total_params:,} parameters")

print("\n2. COMPUTATIONS PER FORWARD PASS (for sequence length L):")
print("   - Encoder embedding lookup: O(L)")
print("   - Encoder LSTM: O(L × H²)")
print("   - Decoder runs L times, each step: O(H²)")
print("   - Decoder FC layer L times: O(L × H × V_tgt)")
print("   - Total: O(L × H² + L × H × V_tgt)")

print("\n" + "="*50)
print("DONE! Model trained and saved.")
print("="*50)