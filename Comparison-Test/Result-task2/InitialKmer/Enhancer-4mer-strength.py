import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import csv
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision.models import resnet18
import math
from itertools import chain
import torch.optim as optim
from stable_baselines3 import PPO
import gym
from gym import spaces
import numpy as np
import random
from sklearn.metrics import matthews_corrcoef, accuracy_score, recall_score, precision_score, roc_auc_score


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

batch_size = 64
embedding_dim = 128
hidden_dim = 128
pretrain_classifier_epochs = 100
joint_train_epochs = 100
kmer = 4  # split kmer
early_stop = 100


class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_accuracy = 0
        self.delta = delta

    def __call__(self, val_accuracy, model, ppo_agent):

        score = val_accuracy

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_accuracy, model, ppo_agent)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_accuracy, model, ppo_agent)
            self.counter = 0

    def save_checkpoint(self, val_accuracy, model, ppo_agent):
        if self.verbose:
            print(f'Validation accuracy increased ({self.best_accuracy:.6f} --> {val_accuracy:.6f}).  Best model ...')
        self.best_accuracy = val_accuracy


class FeatureSelectionEnv(gym.Env):
    def __init__(self, model, criterion, data_loader, device):
        super(FeatureSelectionEnv, self).__init__()
        self.model = model
        self.criterion = criterion
        self.data_loader = data_loader
        self.device = device
        self.batch = None

        self.action_space = spaces.MultiBinary(192)
        self.observation_space = spaces.Box(low=0, high=1, shape=(192,), dtype=np.float32)


    def reset(self):

        self.data_iterator = iter(self.data_loader)
        self.batch = next(self.data_iterator)
        text, labels, lengths = self.batch
        text = text.to(device)
        text = text.permute(1, 0)

        self.model.eval()
        with torch.no_grad():
            _ = self.model(text)  # Forward pass to update internal states
            initial_features, _ = self.model.get_feature_vector()
        return initial_features.mean(dim=0).detach().cpu().numpy()  # Return the mean of features to simplify


    def step(self, action):

        self.model.apply_feature_mask(action)
        text, labels, lengths = self.batch
        text = text.to(device)
        labels = labels.to(device)
        text = text.permute(1, 0)

        self.model.eval() # Make sure model is in eval mode
        with torch.no_grad():
            outputs = self.model(text)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
        reward = correct / labels.size(0)


        try:
            self.batch = next(self.data_iterator)
            next_text, _, _ = self.batch
            next_text = next_text.to(device)
            next_text = next_text.permute(1, 0)
            with torch.no_grad():
                _ = self.model(next_text)  # Update features based on the next batch

            next_features, _ = self.model.get_feature_vector()
            next_features = next_features.mean(dim=0).detach().cpu().numpy()
            done = False
        except StopIteration:
            next_features = np.zeros((192,))  # End of dataset
            done = True  # Signal that the episode is done

        return next_features, reward, done, {}


class SequenceDataset(Dataset):
    def __init__(self, file_path, kmer, vocab=None):
        self.data = self._load_data(file_path, kmer)
        self.vocab = vocab

    def _load_data(self, file_path, kmer):
        data = []
        with open(file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                sequence = self._gain_kmer(row['sequence'], kmer)
                target = int(row['strength'])
                if target != 333:
                    data.append((sequence, target))
        return data

    def _gain_kmer(self, seq, kmer):
        split_sequence = [seq[i:i+kmer] for i in range(len(seq) - kmer + 1)]
        return split_sequence

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence, target = self.data[idx]
        if self.vocab is not None:
            sequence = [self.vocab[token] for token in sequence]
        return sequence, target


def build_vocab(dataset):
    def yield_tokens(data_iter):
        for text, _ in data_iter:
            yield text
    return build_vocab_from_iterator(yield_tokens(dataset), specials=["<unk>", "<pad>"])


def collate_batch(batch, vocab):
    label_list, text_list, lengths = [], [], []
    for _text, _label in batch:
        label_list.append(torch.tensor(_label, dtype=torch.int64))
        processed_text = torch.tensor(_text, dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(processed_text.size(0))
    text_list = pad_sequence(text_list, batch_first=True, padding_value=vocab["<pad>"])
    label_list = torch.tensor(label_list, dtype=torch.int64)
    lengths = torch.tensor(lengths, dtype=torch.int64)
    return text_list, label_list, lengths


def load_dataset(batch_size=32, kmer=3):
    train_data = SequenceDataset("../../../data/train_data.csv", kmer)
    valid_data = SequenceDataset("../../../data/valid_data.csv", kmer)
    test_data = SequenceDataset("../../../data/test_data.csv", kmer)

    vocab = build_vocab(train_data)
    train_data.vocab = vocab
    valid_data.vocab = vocab
    test_data.vocab = vocab

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, \
                              collate_fn=lambda x: collate_batch(x, vocab))
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, \
                              collate_fn=lambda x: collate_batch(x, vocab))
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, \
                             collate_fn=lambda x: collate_batch(x, vocab))

    vocab_size = len(vocab)
    return vocab_size, train_loader, valid_loader, test_loader


class ConvBlock(nn.Module):
    """A basic convolutional block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class ResidualBlock(nn.Module):
    """A residual block for 1D inputs."""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, stride=stride)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual  # Ensure this addition does not use in-place operation
        return F.relu(out, inplace=False)  # Explicitly set inplace to False


class ResNet1D(nn.Module):
    """Simplified ResNet model for 1D sequence data."""
    def __init__(self, in_channels, layers):
        super(ResNet1D, self).__init__()
        self.in_channels = in_channels  # Initialize in_channels
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.in_channels = 32  # Output of conv1

        self.layer1 = self._make_layer(32, layers[0])
        self.layer2 = self._make_layer(64, layers[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, planes, blocks, stride=1):
        layers = []
        downsample = None
        if stride != 1 or self.in_channels != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes)
            )
        for _ in range(blocks):

            layers.append(ResidualBlock(self.in_channels, planes, stride, downsample))
            self.in_channels = planes  # Update the in_channels after the first block
            stride = 1  # Reset stride and downsample after the first block
            downsample = None
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class EnhancerClassifier_v2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, device, num_classes=2, nhead=8, num_layers=3):
        super(EnhancerClassifier_v2, self).__init__()
        self.cnn_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.resnet = ResNet1D(embedding_dim, [2, 2, 2, 2])

        self.transformer_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        transformer_layer  = TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead)
        self.transformer_encoder = TransformerEncoder(transformer_layer, num_layers=num_layers)

        self.fc_1 = nn.Linear(64 + embedding_dim, 64)  # ResNet output + Transformer output
        self.fc_2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        self.feature_mask = torch.ones(192, dtype=torch.float32, device=device)
        self.fused_features = torch.zeros((64, 192))

        self.device = device


    def apply_feature_mask(self, action):
        """Apply the given action as a mask to the feature vector."""
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.bool, device=self.device)
        self.feature_mask = action


    def get_feature_vector(self):
        """Get the feature vector from the last forward pass."""
        return self.fused_features, self.fused_features.shape[0]


    def forward(self, sequence_input):

        # ResNet Pathway
        cnn_embedded = self.cnn_embedding(sequence_input)
        cnn_embedded = cnn_embedded.permute(1, 2, 0)
        resnet_output = self.resnet(cnn_embedded)

        # Transformer Pathway
        transformer_embedded = self.transformer_embedding(sequence_input)
        transformer_embedded = transformer_embedded.permute(1, 0, 2)
        transformer_embedded = self.pos_encoder(transformer_embedded)
        transformer_output = self.transformer_encoder(transformer_embedded)
        transformer_output = transformer_output.mean(dim=1)  # Mean pooling of transformer outputs

        # Concatenate CNN and Transformer outputs
        self.fused_features = torch.cat((resnet_output, transformer_output), dim=1)
        masked_features = self.fused_features * self.feature_mask   # Apply feature mask
        output = self.dropout(masked_features)
        output = self.fc_1(output)
        output = self.relu(output)
        output = self.fc_2(output)

        return output


# Initialization and pre-training of EnhancerClassifier
vocab_size, train_iter, valid_iter, test_iter = load_dataset(batch_size=batch_size, kmer=kmer)
model = EnhancerClassifier_v2(vocab_size, embedding_dim, device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
criterion = nn.CrossEntropyLoss()


def train_classifier(model, iterator, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in iterator:
        text, labels, lengths = batch
        text = text.to(device)
        labels = labels.to(device)
        text = text.permute(1, 0)

        optimizer.zero_grad()
        predictions = model(text)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(iterator)


def evaluate(model, ppo_agent, iterator, criterion):
    model.eval()
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for batch in iterator:
            text, labels, lengths = batch
            text = text.to(device)
            labels = labels.to(device)
            text = text.permute(1, 0)

            _ = model(text)

            # Get the current feature vector for the PPO decision
            current_features, _ = model.get_feature_vector()
            current_features = current_features.mean(dim=0).detach().cpu().numpy()
            # PPO agent decides which features to keep (action)
            action, _states = ppo_agent.predict(current_features, deterministic=True)
            # Apply the action (feature mask) to the classifier
            model.apply_feature_mask(action)

            # Do another forward pass with the masked features
            predictions = model(text)
            probs = F.softmax(predictions, dim=1)[:, 1]
            _, predicted = torch.max(predictions.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = 100 * correct / total
    mcc = matthews_corrcoef(all_labels, all_predictions)
    sn = recall_score(all_labels, all_predictions, pos_label=1)
    sp = recall_score(all_labels, all_predictions, pos_label=0)
    auc = roc_auc_score(all_labels, all_probs)

    return accuracy, mcc, sn, sp, auc


# pretrain the classifier
for epoch in range(pretrain_classifier_epochs):

    print('pretrain classifier epochs: ', epoch)
    loss = train_classifier(model, train_iter, optimizer, criterion, device)


# Instantiate and pre-train the PPO agent
env = FeatureSelectionEnv(model, criterion, train_iter, device)
model_ppo = PPO("MlpPolicy", env, verbose=1)
# Train PPO for a single episode to test
model_ppo.learn(total_timesteps=len(train_iter))


def joint_training(model, ppo_agent, optimizer, criterion, scheduler, epochs, \
                   train_loader, val_loader, test_loader, device, patience=20):
    print()
    print(' *** jointly training *** ')

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(epochs):
        total_loss = 0
        total_accuracy = 0
        count_batches = 0

        for batch in train_loader:  # train_loader
            text, labels, lengths = batch
            text = text.to(device)
            labels = labels.to(device)
            text = text.permute(1, 0)

            # Get current feature vector and make a prediction
            optimizer.zero_grad()
            outputs = model(text)
            loss = criterion(outputs, labels)
            loss.backward()

            # Get the current feature vector for the PPO decision
            current_features, _ = model.get_feature_vector()
            current_features = current_features.mean(dim=0).detach().cpu().numpy()
            # PPO agent decides which features to keep (action)
            action, _states = ppo_agent.predict(current_features, deterministic=True)

            # Apply the action (feature mask) to the classifier
            model.apply_feature_mask(action)

            predictions = model(text)
            modified_loss = criterion(predictions, labels)
            modified_loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(predictions.data, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / labels.size(0)

            total_loss += modified_loss.item()
            total_accuracy += accuracy
            count_batches += 1


        # Provide feedback to PPO agent about the performance
        ppo_agent.learn(total_timesteps=len(train_loader), reset_num_timesteps=True)

        scheduler.step()  # Adjust learning rate after each epoch

        average_loss = total_loss / count_batches
        average_accuracy = total_accuracy / count_batches
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.2f}, Accuracy: {average_accuracy:.2f}%')

        # Evaluate on validation set
        val_accuracy, _, _, _, _ = evaluate(model, ppo_agent, val_loader, criterion)
        print(f'Epoch {epoch + 1}/{epochs}, Val Accuracy: {val_accuracy:.2f}%')

        # Early stopping
        early_stopping(val_accuracy, model, ppo_agent)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Evaluate performance on the test set
    test_acc, test_mcc, test_sn, test_sp, test_auc = evaluate(model, ppo_agent, test_loader, criterion)
    print(f'Best model Test Accuracy: {test_acc:.6f}%, Test MCC: {test_mcc:.6f}, \
            Test Sn: {test_sn:.6f}, Test Sp: {test_sp:.6f}, Test AUC: {test_auc:.6f}')


# Joint training call
joint_training(model, model_ppo, optimizer, criterion, scheduler, joint_train_epochs, \
               train_iter, valid_iter, test_iter, device, early_stop)


