# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import models, util, datasets, evaluation, losses

from tqdm import tqdm

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
                sentences.append(line)
    return sentences

def train():
    model = load_model(model_name='Langboat/mengzi-bert-base-fin')
    sentences = load_sentence(filepath='../input/cls.txt')
    model_output_path = '../model/tsdae'

    train_dataset = datasets.DenoisingAutoEncoderDataset(sentences)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path='Langboat/mengzi-bert-base-fin', tie_encoder_decoder=True)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=10,
        weight_decay=0,
        scheduler='constantlr',
        optimizer_params={'lr': 3e-5},
        show_progress_bar=True,
        checkpoint_path=model_output_path,
        use_amp=True                #Set to True, if your GPU supports FP16 cores
    )

if __name__ == '__main__':
    train()
