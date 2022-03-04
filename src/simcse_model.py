# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import models, util, datasets, evaluation, losses

from tqdm import tqdm
import math


def load_model(model_name='Langboat/mengzi-bert-base-fin', max_seq_length=128):
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'cls')
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model

def load_sentence(filepath=''):
    sentences = []
    with open(filepath, encoding='utf8') as fIn:
        for line in tqdm.tqdm(fIn, desc='Read file'):
            line = line.strip()
            if len(line) >= 10:
                sentences.append([line, line])
    return sentences

def train():
    model = load_model(model_name='Langboat/mengzi-bert-base-fin')
    sentences = load_sentence(filepath='../input/cls.txt')
    model_output_path = '../model/simcse'
    num_epochs = 5
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)

    train_dataset = datasets.DenoisingAutoEncoderDataset(sentences)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          optimizer_params={'lr': 5e-5},
          checkpoint_path=model_output_path,
          show_progress_bar=True,
          use_amp=True  # Set to True, if your GPU supports FP16 cores
    )

if __name__ == '__main__':
    train()
