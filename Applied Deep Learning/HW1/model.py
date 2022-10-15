from typing import Dict

import torch
from torch.nn import Embedding, GRU, LSTM, Sequential, Dropout, Linear, BatchNorm1d
import utils


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        forward_method: str = "basic",
        model_out: str = "output",
        ruduce_seq_method: str = "average",
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.ruduce_seq_method = ruduce_seq_method
        embed_dim = embeddings.size(1)
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.forward_method = forward_method
        self.model_out = model_out
        self.gru = GRU(
            embed_dim,
            self.hidden_size,
            num_layers,
            dropout=dropout,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        self.classifier = Sequential(Dropout(dropout), Linear(self.encoder_output_size, num_class))
        self.batchnorm = BatchNorm1d(self.encoder_output_size)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        if self.bidirectional == True:
            output_size = self.hidden_size * 2
        else:
            output_size = self.hidden_size
        return output_size
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward

        # [batch_size, seq_len]
        text_idx = batch["text_idx"]
        # [batch_size, seq_len, embed_dim]
        embedding = self.embed(text_idx)

        if self.forward_method == "pad_pack":
            # [non-padding numbers, embed_dim]
            packed_embedding = torch.nn.utils.rnn.pack_padded_sequence(embedding, batch["len"], batch_first=True)

            # faster
            self.gru.flatten_parameters()

            # [non-padding numbers, hid_dim * n_dir], [n_layers * n_dir, batch_size, hid_dim]
            packed_encode, h = self.gru(packed_embedding)

            # [batch_size, seq_len, hid_dim * n_dir]
            encode, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_encode, batch_first=True)

        else:
            # [batch_size, seq_len, hid_dim * n_dir], [n_layers * n_dir, batch_size, hid_dim]
            encode, h = self.gru(embedding)

        # [batch_size, hid_dim * n_dir]
        if self.forward_method == "pad_pack" or self.model_out == "hidden":
            if self.bidirectional:
                h = torch.cat((h[-1], h[-2]), axis=-1)
            else:
                h = h[-1]
            # [batch_size, num_class]
            batch["logits"] = self.classifier(h)

        else:
            # [batch_size, hid_dim * n_dir]
            if self.ruduce_seq_method == "average":
                last_seq_encode = self.batchnorm(encode[:, -1])
            else:
                last_seq_encode = encode[:, -1, :]
            # [batch_size, num_class]
            batch["logits"] = self.classifier(last_seq_encode)

        return batch


class SeqTagger(SeqClassifier):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward

        # [batch_size, seq_len]
        tokens_idx = batch["tokens_idx"]

        # [batch_size, seq_len, embed_dim]
        embedding = self.embed(tokens_idx)
        if self.forward_method == "pad_pack":
            # [non-padding numbers, embed_dim]
            packed_embedding = torch.nn.utils.rnn.pack_padded_sequence(embedding, batch["len"], batch_first=True)

            # faster
            self.gru.flatten_parameters()

            # [non-padding numbers, hid_dim * n_dir], [n_layers * n_dir, batch_size, hid_dim]
            packed_encode, h = self.gru(packed_embedding)

            # [batch_size, seq_len, hid_dim * n_dir]
            encode, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_encode, batch_first=True)

        else:
            # [batch_size, seq_len, hid_dim * n_dir], [n_layers * n_dir, batch_size, hid_dim]
            encode, h = self.gru(embedding)

        # [batch_size, seq_len, num_class]
        logits = self.classifier(encode)

        # batch["mask"] = batch["mask"][:, : logits.size(1)]
        # batch["tags_idx"] = batch["tags_idx"][:, : logits.size(1)]

        # [batch_size, num_class, seq_len]
        batch["logits"] = logits.permute(0, 2, 1)

        return batch
        raise NotImplementedError
