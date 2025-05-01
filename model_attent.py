import torch
import torch.nn as nn
import math
from config import NUM_CLASSES, DROPOUT_RATE

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=16):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x has shape [batch_size, seq_len, d_model]
        position_encoding = self.pe[:x.size(1), :]  # Get seq_len positions
        return self.dropout(x + position_encoding)

class AttentionNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(AttentionNet, self).__init__()
        
        # Feature extractors (convolutional layers from AlexNet)
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            # Layer 2
            nn.Conv2d(64, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            # Layer 3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Layer 4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Layer 5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # Constants for the transformer part
        self.d_model = 512  # Transformer dimension
        self.nhead = 8     # Number of attention heads
        self.num_tokens = 16  # 4x4 feature map = 16 tokens
        
        # Projection layer from CNN features to transformer dimension
        self.projection = nn.Linear(256, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, DROPOUT_RATE)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=2048,
            dropout=DROPOUT_RATE,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, 512),
            nn.ReLU(True),
            nn.Dropout(p=DROPOUT_RATE),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def forward(self, x):
        # Extract features using CNN layers (B, 256, 4, 4)
        x = self.features(x)
        
        # Reshape to (B, 16, 256) - each 256-dim vector represents a spatial location
        batch_size = x.size(0)
        x = x.view(batch_size, 256, -1).transpose(1, 2)  # (B, 16, 256)
        
        # Project to transformer dimension
        x = self.projection(x)  # (B, 16, 512)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling over token dimension
        x = torch.mean(x, dim=1)  # (B, 512)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
