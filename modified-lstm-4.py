# _*_ encoding:utf-8 _*_
__author__ = 'Cai Jinwang'
__date__ = '2024/9/13 15:45'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import jieba
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score
import time  # 引入time模块

# 确认库版本
print("PyTorch version:", torch.__version__)
print("Numpy version:", np.__version__)
print("Jieba version:", jieba.__version__)

# 读取文件函数
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return lines

# 读取数据
train_lines = read_file('./data/train.txt')
test_lines = read_file('./data/test.txt')
dev_lines = read_file('./data/dev.txt')
class_lines = read_file('./data/class.txt')

# 分词函数
def tokenize_text(text):
    return ' '.join(jieba.cut(text))

# 提取文本和标签，同时添加检查以防止索引错误
train_texts, train_labels = [], []
for line in train_lines:
    parts = line.split('\t')
    if len(parts) >= 2:  # 确保行中有至少两个元素（文本和标签）
        train_texts.append(tokenize_text(parts[0]))
        train_labels.append(parts[1].strip())
    else:
        print(f"Warning: Skipping malformed line: {line.strip()}")  # 输出提示跳过格式不正确的行

# 对测试集和开发集进行相同处理
test_texts, test_labels = [], []
for line in test_lines:
    parts = line.split('\t')
    if len(parts) >= 2:
        test_texts.append(tokenize_text(parts[0]))
        test_labels.append(parts[1].strip())
    else:
        print(f"Warning: Skipping malformed line: {line.strip()}")

dev_texts, dev_labels = [], []
for line in dev_lines:
    parts = line.split('\t')
    if len(parts) >= 2:
        dev_texts.append(tokenize_text(parts[0]))
        dev_labels.append(parts[1].strip())
    else:
        print(f"Warning: Skipping malformed line: {line.strip()}")


# 使用LabelEncoder进行标签编码
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)
dev_labels_encoded = label_encoder.transform(dev_labels)

# 构建词汇表
vocab = set(' '.join(train_texts).split())
vocab_size = len(vocab)
word_to_ix = {word: i for i, word in enumerate(vocab)}

# 文本转换为索引序列
def text_to_sequence(text):
    sequence = [word_to_ix[word] for word in text.split() if word in word_to_ix]
    return sequence

# 创建数据集类
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        sequence = torch.tensor(text_to_sequence(text), dtype=torch.long)
        return sequence, label

# 自定义的 collate_fn 函数用于在批处理时进行填充
def collate_fn(batch):
    sequences, labels = zip(*batch)
    # 使用 pad_sequence 将所有序列填充到批次中最长的序列长度
    padded_sequences = pad_sequence([torch.tensor(seq).clone().detach().long() for seq in sequences],
                                    batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_sequences, labels

# 检查是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 数据集和加载器
train_dataset = TextDataset(train_texts, train_labels_encoded)
test_dataset = TextDataset(test_texts, test_labels_encoded)
dev_dataset = TextDataset(dev_texts, dev_labels_encoded)

# 修改 DataLoader，添加 collate_fn
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
dev_loader = DataLoader(dev_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

# 定义LSTM模型
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))  # [batch_size, seq_len, embedding_dim]
        output, (hidden, cell) = self.lstm(embedded)  # output: [batch_size, seq_len, hidden_dim]
        if self.lstm.bidirectional:
            hidden = self.dropout(
                torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))  # hidden: [batch_size, hidden_dim * 2]
        else:
            hidden = self.dropout(hidden[-1, :, :])  # hidden: [batch_size, hidden_dim]
        return self.fc(hidden)  # [batch_size, output_dim]

# 实例化模型并将其移动到设备（GPU或CPU）
model = LSTMClassifier(vocab_size, embedding_dim=100, hidden_dim=128, output_dim=len(label_encoder.classes_),
                       n_layers=2, bidirectional=True, dropout=0.5).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 训练模型
def train(model, iterator, optimizer, criterion, epoch_num):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    batch_num = 0

    start_time = time.time()  # 开始记录epoch时间
    for sequences, labels in iterator:
        batch_start_time = time.time()  # 开始记录batch时间

        # 将数据移动到GPU
        sequences, labels = sequences.to(device), labels.to(device)

        # 前向传播
        predictions = model(sequences)

        # 计算损失
        loss = criterion(predictions, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算准确率
        acc = (predictions.argmax(1) == labels).float().mean()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
        batch_num += 1

        batch_end_time = time.time()  # 记录batch结束时间
        print(f'Epoch {epoch_num + 1}, Batch {batch_num}, Batch Size: {len(sequences)}, Batch Time: {batch_end_time - batch_start_time:.2f}s')

    epoch_end_time = time.time()  # 记录epoch结束时间
    print(f'Epoch {epoch_num + 1}, Total Epoch Time: {epoch_end_time - start_time:.2f}s')

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# 评估模型
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for sequences, labels in iterator:
            sequences, labels = sequences.to(device), labels.to(device)

            predictions = model(sequences)
            loss = criterion(predictions, labels)
            acc = (predictions.argmax(1) == labels).float().mean()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predictions.argmax(1).cpu().numpy())

    # 计算每个类别的准确率、召回率、F1分数
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# 训练和评估
num_epochs = 10
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch)
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(
        f'Epoch: {epoch + 1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc * 100:.2f}%, Test Loss: {test_loss:.3f}, Test Acc: {test_acc * 100:.2f}%')

# 保存模型
torch.save(model.state_dict(), 'lstm_text_classification_model.pth')

# 加载模型
model.load_state_dict(torch.load('lstm_text_classification_model.pth'))

# 进行预测
def predict_text(text, model, word_to_ix):
    model.eval()
    sequence = torch.tensor([word_to_ix[word] for word in text.split() if word in word_to_ix],
                            dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(sequence)
    predicted_class_index = prediction.argmax(1).item()
    predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
    return predicted_class

# 测试预测功能
test_text = "这是一个测试文本。"
predicted_label = predict_text(tokenize_text(test_text), model, word_to_ix)
print(f'The predicted label for the test text is: {predicted_label}')