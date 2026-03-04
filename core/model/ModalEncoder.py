import torch
from torch import nn
import torch.nn.functional as F


class MetricEncoder(nn.Module):
    """Time-series metric encoder: 1D CNN for sequential metrics."""
    def __init__(self, input_channels=12, seq_len=20, output_dim=128):
        super(MetricEncoder, self).__init__()
        # Input: [batch, 20, 12] -> [batch, 12, 20]
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, output_dim)
        
    def forward(self, x):
        # x: [batch, seq_len, input_channels]
        x = x.transpose(1, 2)  # [batch, input_channels, seq_len]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)  # [batch, 128]
        x = self.fc(x)  # [batch, output_dim]
        return x


class LogEncoder(nn.Module):
    """Log template encoder: MLP over template statistics."""
    def __init__(self, input_dim=48, output_dim=128):
        super(LogEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        # x: [batch, input_dim]
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)  # [batch, output_dim]
        return x


class TraceEncoder(nn.Module):
    """
    Two-channel input: Duration + Error Rate
    """
    def __init__(self, seq_len=20, output_dim=128):
        super(TraceEncoder, self).__init__()
        # Input: [batch, 20, 2]
        # Channel 0: Duration (Normalized)
        # Channel 1: Error Rate (0-1)
        self.conv1 = nn.Conv1d(2, 32, kernel_size=3, padding=1)  # input_channels=2
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, output_dim)
        
    def forward(self, x):
        # x: [batch, seq_len, 2]
        x = x.transpose(1, 2)  # [batch, 2, seq_len]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)  # [batch, 64]
        x = self.fc(x)  # [batch, output_dim]
        return x


class MultiModalEncoder(nn.Module):
    """
    Multimodal encoder.
    Encodes raw time-series into fixed-dimensional embeddings, then feeds them into
    a graph network for fault diagnosis.
    """
    def __init__(self, output_dim=128, metric_channels=12, log_dim=48, seq_len=20):
        super(MultiModalEncoder, self).__init__()
        self.metric_encoder = MetricEncoder(
            input_channels=metric_channels, 
            seq_len=seq_len, 
            output_dim=output_dim
        )
        self.log_encoder = LogEncoder(input_dim=log_dim, output_dim=output_dim)
        self.trace_encoder = TraceEncoder(seq_len=seq_len, output_dim=output_dim)
        
    def forward(self, metric_data, log_data, trace_data):
        """
        Args:
            metric_data: [batch, seq_len, metric_channels]
            log_data: [batch, log_dim]
            trace_data: [batch, seq_len, 2]
        Returns:
            metric_emb: [batch, output_dim]
            log_emb: [batch, output_dim]
            trace_emb: [batch, output_dim]
        """
        metric_emb = self.metric_encoder(metric_data)
        log_emb = self.log_encoder(log_data)
        trace_emb = self.trace_encoder(trace_data)
        return metric_emb, log_emb, trace_emb
