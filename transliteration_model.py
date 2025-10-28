import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
# 2. IMPROVED MODEL WITH ATTENTION
# ============================================

class Attention(nn.Module):
    """Bahdanau Attention Mechanism"""
    def __init__(self, decoder_hidden_dim, encoder_hidden_dim):
        super(Attention, self).__init__()
        self.decoder_hidden_dim = decoder_hidden_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.attn = nn.Linear(decoder_hidden_dim + encoder_hidden_dim, decoder_hidden_dim)
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        # hidden: (num_layers, batch_size, hidden_dim)
        # encoder_outputs: (batch_size, src_len, hidden_dim)
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Use the last layer's hidden state
        hidden = hidden[-1]  # (batch_size, hidden_dim)
        
        # Repeat hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # (batch_size, src_len, hidden_dim)
        
        # Concatenate hidden and encoder_outputs
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy: (batch_size, src_len, hidden_dim)
        
        attention = self.v(energy).squeeze(2)  # (batch_size, src_len)
        
        return F.softmax(attention, dim=1)


class Encoder(nn.Module):
    """Improved Encoder with Bidirectional LSTM"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, 
                 cell_type='LSTM', dropout=0.3):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        if cell_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                              batch_first=True, dropout=dropout if num_layers > 1 else 0,
                              bidirectional=True)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers, 
                             batch_first=True, dropout=dropout if num_layers > 1 else 0,
                             bidirectional=True)
        else:  # RNN
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, 
                             batch_first=True, dropout=dropout if num_layers > 1 else 0,
                             bidirectional=True)
        
        # Linear layer to convert bidirectional outputs to single direction
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.dropout(self.embedding(x))  # (batch_size, seq_len, embedding_dim)
        
        if self.cell_type == 'LSTM':
            outputs, (hidden, cell) = self.rnn(embedded)
            # outputs: (batch_size, seq_len, hidden_dim*2)
            # hidden, cell: (num_layers*2, batch_size, hidden_dim)
            
            # Combine forward and backward hidden states
            hidden = torch.tanh(self.fc_hidden(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
            cell = torch.tanh(self.fc_cell(torch.cat((cell[-2,:,:], cell[-1,:,:]), dim=1)))
            
            # Reshape to (num_layers, batch_size, hidden_dim)
            hidden = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
            cell = cell.unsqueeze(0).repeat(self.num_layers, 1, 1)
            
            return outputs, (hidden, cell)
        else:
            outputs, hidden = self.rnn(embedded)
            hidden = torch.tanh(self.fc_hidden(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
            hidden = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
            return outputs, hidden


class Decoder(nn.Module):
    """Improved Decoder with Attention"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, encoder_hidden_dim, num_layers=2, 
                 cell_type='LSTM', dropout=0.3):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(hidden_dim)
        
        # Input to RNN is embedding + context (encoder outputs are bidirectional: encoder_hidden_dim * 2)
        if cell_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim + encoder_hidden_dim, hidden_dim, num_layers, 
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif cell_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim + encoder_hidden_dim, hidden_dim, num_layers, 
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:  # RNN
            self.rnn = nn.RNN(embedding_dim + encoder_hidden_dim, hidden_dim, num_layers, 
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Output layer: hidden + context + embedding -> vocab
        self.fc = nn.Linear(hidden_dim + encoder_hidden_dim + embedding_dim, vocab_size)
    
    def forward(self, x, hidden, encoder_outputs):
        # x: (batch_size, 1)
        # hidden: (num_layers, batch_size, hidden_dim) or tuple for LSTM
        # encoder_outputs: (batch_size, src_len, encoder_hidden_dim*2)
        
        embedded = self.dropout(self.embedding(x))  # (batch_size, 1, embedding_dim)
        
        # Calculate attention weights
        if self.cell_type == 'LSTM':
            a = self.attention(hidden[0], encoder_outputs)  # (batch_size, src_len)
        else:
            a = self.attention(hidden, encoder_outputs)
        
        a = a.unsqueeze(1)  # (batch_size, 1, src_len)
        
        # Calculate weighted context vector
        context = torch.bmm(a, encoder_outputs)  # (batch_size, 1, encoder_hidden_dim*2)
        
        # Concatenate embedded and context
        rnn_input = torch.cat((embedded, context), dim=2)  # (batch_size, 1, embedding_dim + encoder_hidden_dim*2)
        
        if self.cell_type == 'LSTM':
            output, (hidden, cell) = self.rnn(rnn_input, hidden)
            # Concatenate output, context, and embedded for prediction
            prediction = self.fc(torch.cat((output.squeeze(1), context.squeeze(1), embedded.squeeze(1)), dim=1))
            return prediction, (hidden, cell), a.squeeze(1)
        else:
            output, hidden = self.rnn(rnn_input, hidden)
            prediction = self.fc(torch.cat((output.squeeze(1), context.squeeze(1), embedded.squeeze(1)), dim=1))
            return prediction, hidden, a.squeeze(1)


class Seq2SeqWithAttention(nn.Module):
    """Sequence to Sequence Model with Attention"""
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqWithAttention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # src: (batch_size, src_len)
        # tgt: (batch_size, tgt_len)
        
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.vocab_size
        
        # Store outputs and attention weights
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        attentions = torch.zeros(batch_size, tgt_len, src.shape[1]).to(self.device)
        
        # Encode the source sequence
        encoder_outputs, encoder_hidden = self.encoder(src)
        
        # First input to decoder is <SOS> token
        decoder_input = tgt[:, 0].unsqueeze(1)  # (batch_size, 1)
        decoder_hidden = encoder_hidden
        
        # Decode one character at a time
        for t in range(1, tgt_len):
            decoder_output, decoder_hidden, attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            outputs[:, t, :] = decoder_output
            attentions[:, t, :] = attention
            
            # Teacher forcing: use actual target as next input
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = decoder_output.argmax(1)
            decoder_input = tgt[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        return outputs, attentions
    
    def predict(self, src, max_len=50):
        """Predict without teacher forcing"""
        self.eval()
        with torch.no_grad():
            batch_size = src.shape[0]
            
            # Encode
            encoder_outputs, encoder_hidden = self.encoder(src)
            
            # Start with <SOS>
            decoder_input = torch.tensor([[1]] * batch_size).to(self.device)  # <SOS>
            decoder_hidden = encoder_hidden
            
            predictions = []
            attentions = []
            
            for _ in range(max_len):
                decoder_output, decoder_hidden, attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                top1 = decoder_output.argmax(1)
                predictions.append(top1.unsqueeze(1))
                attentions.append(attention)
                
                decoder_input = top1.unsqueeze(1)
                
                # Stop if all sequences predict <EOS>
                if (top1 == 2).all():  # <EOS> token
                    break
            
            return torch.cat(predictions, dim=1), torch.stack(attentions, dim=1)


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
        
        output, _ = model(src, tgt)
        
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
            
            output, _ = model(src, tgt, teacher_forcing_ratio=0)  # No teacher forcing
            
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
    predictions, attentions = model.predict(src_tensor, max_len)
    
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

# IMPROVED CONFIGURATION
CONFIG = {
    'embedding_dim': 256,      # Increased from 128
    'hidden_dim': 512,         # Increased from 256
    'num_layers': 2,           # Increased from 1
    'cell_type': 'LSTM',
    'dropout': 0.3,            # Increased from 0.1
    'batch_size': 64,
    'learning_rate': 0.001,
    'num_epochs': 25,          # Increased from 10
    'clip': 1,
    'teacher_forcing_ratio': 0.5,
}

print("="*60)
print("🚀 IMPROVED CONFIGURATION WITH ATTENTION")
print("="*60)
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# Load data
print("\n" + "="*60)
print("LOADING DATA")
print("="*60)

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

# Initialize improved model
print("\n" + "="*60)
print("INITIALIZING IMPROVED MODEL WITH ATTENTION")
print("="*60)

encoder = Encoder(len(src_vocab), CONFIG['embedding_dim'], CONFIG['hidden_dim'], 
                 CONFIG['num_layers'], CONFIG['cell_type'], CONFIG['dropout'])
decoder = Decoder(len(tgt_vocab), CONFIG['embedding_dim'], CONFIG['hidden_dim'], 
                 CONFIG['hidden_dim'] * 2, CONFIG['num_layers'], CONFIG['cell_type'], CONFIG['dropout'])

model = Seq2SeqWithAttention(encoder, decoder, device).to(device)

# Count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total trainable parameters: {count_parameters(model):,}")
print("✅ Model now includes:")
print("   • Bidirectional Encoder")
print("   • Bahdanau Attention Mechanism")
print("   • Deeper Network (2 layers)")
print("   • Higher Capacity (512 hidden units)")

# Initialize optimizer and loss with learning rate scheduler
optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                  factor=0.5, patience=2)
criterion = nn.CrossEntropyLoss(ignore_index=0)
print("Scheduler: ReduceLROnPlateau initialized (reduces LR by 0.5 if no improvement for 2 epochs)")

# Training loop
print("\n" + "="*60)
print("TRAINING WITH IMPROVED ARCHITECTURE")
print("="*60)

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
    
    # Update learning rate
    scheduler.step(valid_loss)
    
    # Save best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'best_model_attention.pt')
        print("  → Best model saved!")

# Plot losses
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', linewidth=2)
plt.plot(valid_losses, label='Valid Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
epochs = range(1, len(train_losses) + 1)
plt.plot(epochs, train_losses, 'o-', label='Train Loss', linewidth=2, markersize=4)
plt.plot(epochs, valid_losses, 's-', label='Valid Loss', linewidth=2, markersize=4)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.title('Loss Curves (Detailed)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Load best model for testing
model.load_state_dict(torch.load('best_model_attention.pt'))

# Test the model
print("\n" + "="*60)
print("TESTING IMPROVED MODEL")
print("="*60)

test_examples = ['ghar', 'dost', 'paani', 'kitaab', 'pyaar', 'ajanabee', 
                 'namaste', 'dhanyavaad', 'bharat', 'school']

print("\nTransliteration Examples:")
for word in test_examples:
    prediction = transliterate(model, word, src_vocab, tgt_vocab, device)
    print(f"  {word:15} -> {prediction}")

# Calculate test loss
test_loss = evaluate(model, test_loader, criterion, device)
print(f"\nTest Loss: {test_loss:.4f}")

print("\n" + "="*60)
print("IMPROVEMENTS SUMMARY")
print("="*60)
print("\n✅ What's New:")
print("   1. Bidirectional Encoder - captures context from both directions")
print("   2. Attention Mechanism - focuses on relevant input characters")
print("   3. Deeper Network - 2 layers for more capacity")
print("   4. Larger Hidden Dimension - 512 units (was 256)")
print("   5. Better Regularization - 0.3 dropout (was 0.1)")
print("   6. Learning Rate Scheduling - adapts LR during training")
print("   7. More Training - 25 epochs (was 10)")
print("\n🎯 Expected Improvement: 45-65% word accuracy (was ~29%)")

print("\n" + "="*60)
print("DONE! Improved model trained and saved.")
print("="*60)
