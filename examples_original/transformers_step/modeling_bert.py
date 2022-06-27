# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model. """

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import math
import os
import sys
from io import open

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from .modeling_utils import PreTrainedModel, prune_linear_layer
from .configuration_bert import BertConfig
from .file_utils import add_start_docstrings
from .encoder import TransformerInterEncoder
import torch.nn.functional as F
from .generator import Beam

logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    'bert-base-german-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
}

def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error("Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new}


BertLayerNorm = torch.nn.LayerNorm

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)  # [30522, 768] 
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)  # [512, 768]  wo do not to edit this 
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)  # [2, 768]

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)  # seq_length=512
        if position_ids is None:  
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # [0, 1, ..., 511]
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # [[0, 1, ..., 511]] (1, 512) -> [batch_size, seq_length]
            # we should modify this, to input position_ids 

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
            # we should modify this, to input token_type_ids 

        words_embeddings = self.word_embeddings(input_ids)   # input_ids [batch_size, seq_length] 
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_tensor, attention_mask=None, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)

        return outputs


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.Mydense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.Mydense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


BERT_START_DOCSTRING = r"""    The BERT model was proposed in
    `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_
    by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. It's a bidirectional transformer
    pre-trained using a combination of masked language modeling objective and next sentence prediction
    on a large corpus comprising the Toronto Book Corpus and Wikipedia.

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`:
        https://arxiv.org/abs/1810.04805

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model. 
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            To match pre-training, BERT input sequence should be formatted with [CLS] and [SEP] tokens as follows:

            (a) For sequence pairs:

                ``tokens:         [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]``
                
                ``token_type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1``

            (b) For single sequences:

                ``tokens:         [CLS] the dog is hairy . [SEP]``
                
                ``token_type_ids:   0   0   0   0  0     0   0``

            Bert is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.

            Indices can be obtained using :class:`transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            (see `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_ for more details).
        **position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
"""

class MLP(nn.Module):
    def __init__(self, input_sizes, dropout_prob=0.2, bias=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(input_sizes)):
            self.layers.append(nn.Linear(input_sizes[i - 1], input_sizes[i], bias=bias))
        self.norm_layers = nn.ModuleList()
        if len(input_sizes) > 2:
            for i in range(1, len(input_sizes) - 1):
                self.norm_layers.append(nn.LayerNorm(input_sizes[i]))
        self.drop_out = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(self.drop_out(x))
            if i < len(self.layers) - 1:
                x = gelu(x)
                if len(self.norm_layers):
                    x = self.norm_layers[i](x)
        return x

@add_start_docstrings("The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
                      BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertModel(BertPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    def __init__(self, config):
        super(BertModel, self).__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]

        pooled_output = sequence_output[:, 0]

        outputs = (sequence_output, pooled_output) + encoder_outputs[1:]
        return outputs


class HierarchicalAttention(nn.Module):
    def __init__(self, config,):
        super(HierarchicalAttention, self).__init__()

        self.linear_in_2 = nn.Linear(config.hidden_size, 1, bias=False)

        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.sentence_tran = nn.Linear(config.hidden_size, config.hidden_size)
        self.sentence_tran_2 = nn.Linear(config.hidden_size, 1)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        ### pairwise loss ###
        self.pairwise_relationship = nn.Linear(config.hidden_size, 2)
        self.h1_relationship = nn.Linear(config.hidden_size, 2)
        self.h2_relationship = nn.Linear(config.hidden_size, 2)


    def forward(self, sequence_output, pairs_list, passage_length, pairs_num, sep_positions, cuda, mask_cls, cls_pooled_output):

        expanded_sample_num, max_len, hidden_size = sequence_output.size()

        batch, max_pair_num, _ = sep_positions.size()
        _, max_num_sen = mask_cls.size()

        sep_positions = sep_positions.reshape(expanded_sample_num, 2)

        top_vec = sequence_output

        top_vec_tran = self.sentence_tran(top_vec)

        top_vec_tran_tan = torch.tanh(top_vec_tran)

        top_vec_tran_score = self.sentence_tran_2(top_vec_tran_tan).squeeze(-1) 

        selected_masks_all = []

        for sep in sep_positions:

            sep_0 = int(sep[0].cpu())
            sep_1 = int(sep[1].cpu())


            select_mask_0 = [0] + [1]*sep_0 + [0]*(max_len-sep_0-1)
            select_mask_1 = [0]*(sep_0+1) + [1]*(sep_1-sep_0) + [0]*(max_len-sep_1-1)

            new_select_mask = [select_mask_0, select_mask_1]
            selected_masks_all.append(new_select_mask)
    

        selected_masks_all = torch.FloatTensor(selected_masks_all).to(int(cuda[-1]))

        attention_mask = selected_masks_all.to(dtype=next(self.parameters()).dtype)
        attention_mask = (1.0 - attention_mask) * -10000.0

        selected_masks_all_t = selected_masks_all.permute(0,2,1)


        select_score_t = selected_masks_all_t*top_vec_tran_score[:, :, None]  
        select_score = select_score_t.permute(0,2,1)

        attention_scores = select_score + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)   

        attention_probs = self.dropout(attention_probs)


        mix = torch.bmm(attention_probs, top_vec)


        mix = mix.reshape(batch, max_pair_num, 2, hidden_size)
        final_seq_matriax = torch.zeros(batch, max_num_sen, hidden_size).to(int(cuda[-1]))

        ### relative score ####
        cls_score = self.pairwise_relationship(cls_pooled_output)
        cls_score_matrix_nn = torch.zeros(batch, max_num_sen, max_num_sen, 2).to(int(cuda[-1]))
        cls_score_batch = cls_score.reshape(batch, max_pair_num, 2)


        ### history score ###
        cls_score_his1 = self.h1_relationship(cls_pooled_output)
        cls_score_matrix_nn_his1 = torch.zeros(batch, max_num_sen, max_num_sen, 2).to(int(cuda[-1]))
        cls_score_batch_his1 = cls_score_his1.reshape(batch, max_pair_num, 2)

        cls_score_his2 = self.h2_relationship(cls_pooled_output)
        cls_score_matrix_nn_his2 = torch.zeros(batch, max_num_sen, max_num_sen, 2).to(int(cuda[-1]))
        cls_score_batch_his2 = cls_score_his2.reshape(batch, max_pair_num, 2)


        ### cls_output_matrix ###
        cls_output_matrix_nn = torch.zeros(batch, max_num_sen, max_num_sen, hidden_size).to(int(cuda[-1]))
        cls_pooled_output_batch = cls_pooled_output.reshape(batch, max_pair_num, hidden_size)

        

        for idx_i, length in enumerate(passage_length):
            current_true_pair_num = pairs_num[idx_i]
            true_pair_list = pairs_list[idx_i][:current_true_pair_num]

            count = [0 for item in range(length)]
            edge_num = int(len(true_pair_list)/length*2)

            sentence_tensor_i = torch.zeros(length, edge_num, hidden_size).to(int(cuda[-1]))
            for idx_j, pair in enumerate(true_pair_list):

                current_place_0 = count[pair[0]]
                sentence_tensor_i[pair[0]][current_place_0]=mix[idx_i][idx_j][0]
                count[pair[0]] = count[pair[0]] + 1

                current_place_1 = count[pair[1]]
                sentence_tensor_i[pair[1]][current_place_1]=mix[idx_i][idx_j][1]
                count[pair[1]] = count[pair[1]] + 1

                ### cls output ###
                cls_output_matrix_nn[idx_i][pair[0]][pair[1]] = cls_pooled_output_batch[idx_i][idx_j]

                ### relative score ###
                cls_score_matrix_nn[idx_i][pair[0]][pair[1]] = cls_score_batch[idx_i][idx_j]

                ### history score ###
                cls_score_matrix_nn_his1[idx_i][pair[0]][pair[1]] = cls_score_batch_his1[idx_i][idx_j]
                cls_score_matrix_nn_his2[idx_i][pair[0]][pair[1]] = cls_score_batch_his2[idx_i][idx_j]



            sample = sentence_tensor_i
            passage_num, edge_num, hidden_size = sample.size()


            query_2 = sample.view(passage_num * edge_num, hidden_size).to(int(cuda[-1]))
            query_2 = self.linear_in_2(query_2)
            query_2 = query_2.reshape(passage_num, edge_num)


            attention_scores = query_2.unsqueeze(dim=1)

            attention_scores = attention_scores.view(passage_num * 1, edge_num)
            attention_weights = self.softmax(attention_scores)
            attention_weights = attention_weights.view(passage_num, 1, edge_num)


            mix_new = torch.bmm(attention_weights, sample).squeeze()

            final_seq_matriax[idx_i][:length] = mix_new


        return final_seq_matriax, cls_output_matrix_nn, cls_score, cls_score_matrix_nn, cls_score_matrix_nn_his1, cls_score_matrix_nn_his2




@add_start_docstrings("""Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. """,
    BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertForOrdering(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    def __init__(self, config, args):
        super(BertForOrdering, self).__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.encoder = TransformerInterEncoder(config.hidden_size, args.ff_size, args.heads, args.para_dropout, args.inter_layers)
        self.key_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.query_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh_linear = nn.Linear(config.hidden_size, 1)

        self.decoder = nn.LSTM(config.hidden_size, config.hidden_size, batch_first=True)

        self.critic = None

        ### pairwise ###
        self.two_level_encoder = HierarchicalAttention(config)


        # ### pairwise loss ###
        self.pairwise_loss_lam = args.pairwise_loss_lam

        ### pairwise decoder ###
        d_pair_posi = config.hidden_size + 2
        self.pw_k = nn.Linear(d_pair_posi * 4, config.hidden_size, False)

        self.init_weights()
        

    def equip(self, critic):
        self.critic = critic

    def rela_encode(self, cls_output_matrix_nn, cls_score_matrix_nn):

        p_direc = F.softmax(cls_score_matrix_nn, -1)

        rela_vec_diret = torch.cat((cls_output_matrix_nn, p_direc), -1)

        return rela_vec_diret

    def history_encode(self, cls_output_matrix_nn, cls_score_matrix_nn_his1, cls_score_matrix_nn_his2):

        p_left1 = F.softmax(cls_score_matrix_nn_his1, -1)
        p_left2 = F.softmax(cls_score_matrix_nn_his2, -1)

        hist_vec_left1 = torch.cat((cls_output_matrix_nn, p_left1), -1)
        hist_vec_left2 = torch.cat((cls_output_matrix_nn, p_left2), -1)

        return hist_vec_left1, hist_vec_left2


    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                pairs_list=None, passage_length=None, pairs_num=None, sep_positions=None, ground_truth=None, 
                mask_cls=None, pairwise_labels=None, cuda=None, head_mask=None):

        document_matrix, _, hcn, original_key, cls_pooled_output, cls_output_matrix_nn, cls_score, cls_score_matrix_nn, cls_score_matrix_nn_his1, cls_score_matrix_nn_his2 = self.encode(input_ids, attention_mask, token_type_ids, pairs_list, passage_length, pairs_num, 
            sep_positions, ground_truth, mask_cls, pairwise_labels, cuda, head_mask)

        num_sen = passage_length
        order = ground_truth


        target = order
        tgt_len = num_sen
        batch, num = target.size()

        tgt_len_less = tgt_len
        target_less = target

        target_mask = torch.zeros_like(target_less).byte()
        pointed_mask_by_target = torch.zeros_like(target).byte()



        ##### relative mask #####
        eye_mask = torch.eye(num).byte().cuda(int(cuda[-1])).unsqueeze(0)

        loss_left1_mask = torch.tril(target.new_ones(num, num), -1).unsqueeze(0).expand(batch, num, num)
        truth_left1 = loss_left1_mask - torch.tril(target.new_ones(num, num), -2).unsqueeze(0)
        rela_mask = torch.ones_like(truth_left1).byte().cuda(int(cuda[-1])) - eye_mask

        for b in range(batch):
            pointed_mask_by_target[b, :tgt_len[b]] = 1
            target_mask[b, :tgt_len_less[b]] = 1

            rela_mask[b, tgt_len[b]:] = 0
            rela_mask[b, :, tgt_len[b]:] = 0


        ###  decoder ####
        dec_inputs = document_matrix[torch.arange(document_matrix.size(0)).unsqueeze(1), target[:, :-1]]
        start = dec_inputs.new_zeros(batch, 1, dec_inputs.size(2))

        dec_inputs = torch.cat((start, dec_inputs), 1)


        ####  relative vector and score  ####
        rela_vec_diret = self.rela_encode(cls_output_matrix_nn, cls_score_matrix_nn) 

        ####  history vector and score  ####
        hist_vec_left1, hist_vec_left2 = self.history_encode(cls_output_matrix_nn, cls_score_matrix_nn, cls_score_matrix_nn)


        dec_outputs = []
        pw_keys = []
        pointed_mask = [rela_mask.new_zeros(batch, 1, num)]


        eye_zeros = torch.ones_like(eye_mask) - eye_mask
        eye_zeros = eye_zeros.unsqueeze(-1)

        for t in range(num):
            if t == 0:
                rela_mask = rela_mask.unsqueeze(-1)
                l1_mask = torch.zeros_like(rela_mask)
                l2_mask = torch.zeros_like(rela_mask)

            else:
                tar = target[:, t - 1]

                rela_mask[torch.arange(batch), tar] = 0
                rela_mask[torch.arange(batch), :, tar] = 0

                l1_mask = torch.zeros_like(rela_mask)
                l2_mask = torch.zeros_like(rela_mask)

                l1_mask[torch.arange(batch), tar, :] = 1   

                if t > 1:
                    l2_mask[torch.arange(batch), target[:, t - 2], :] = 1    

                pm = pointed_mask[-1].clone().detach()
                pm[torch.arange(batch), :, tar] = 1
                pointed_mask.append(pm)

            cur_hist_l1 = hist_vec_left1.masked_fill(l1_mask == 0, 0).sum(1)  

            cur_hist_l2 = hist_vec_left2.masked_fill(l2_mask == 0, 0).sum(1)


            rela_vec_diret.masked_fill_(rela_mask == 0, 0)

            forw_pw = rela_vec_diret.mean(2) 
            back_pw = rela_vec_diret.mean(1)


            pw_info = torch.cat((cur_hist_l1, cur_hist_l2, forw_pw, back_pw), -1)
            pw_key = self.pw_k(pw_info)
            pw_keys.append(pw_key.unsqueeze(1))

            dec_inp = dec_inputs[:, t:t + 1]


            output, hcn = self.decoder(dec_inp, hcn)
            dec_outputs.append(output)


        dec_outputs = torch.cat(dec_outputs, 1)

        query = self.query_linear(dec_outputs).unsqueeze(2)  # B N 1 H


        original_key = original_key.unsqueeze(1)     


        key = torch.cat(pw_keys, 1)


        e = torch.tanh(query + key + original_key)   
        

        e = self.tanh_linear(e).squeeze(-1)  


        pointed_mask = torch.cat(pointed_mask, 1)
        pointed_mask_by_target = pointed_mask_by_target.unsqueeze(1).expand_as(pointed_mask)

        e.masked_fill_(pointed_mask == 1, -1e9)
        e.masked_fill_(pointed_mask_by_target == 0, -1e9)


        logp = F.log_softmax(e, dim=-1)

        logp = logp.view(-1, logp.size(-1))


        loss = self.critic(logp, target.contiguous().view(-1))

        target_mask = target_mask.view(-1)
        loss.masked_fill_(target_mask == 0, 0)


        transform_loss = loss.reshape(batch, num)

        transform_loss_sample = torch.sum(transform_loss, -1)/(num_sen.float() + 1e-20 - 1)  

        new_original_loss = transform_loss_sample.sum()/batch
        


        ### pairwise loss ok ###
        batch, max_pair_num = pairwise_labels.size()
        pairwise_labels_new = pairwise_labels.reshape(batch*max_pair_num)

        logp_cls = F.log_softmax(cls_score, dim=-1) 

        pairwise_loss = self.critic(logp_cls.view(-1, 2), pairwise_labels_new)


        pair_masks = []
        for item in range(batch):
            pair_mask = [1]* int(pairs_num[item].cpu()) + [0]*int((max_pair_num-pairs_num[item]).cpu())
            pair_masks.append(pair_mask)

        pair_masks = torch.FloatTensor(pair_masks).to(int(cuda[-1]))  # [batch_size, max_pair_num]
        pair_masks_new = pair_masks.view(-1) # [batch_size*max_pair_num]


        pairwise_loss_clean = pairwise_loss*pair_masks_new 

        pairwise_loss_clean = pairwise_loss_clean.reshape(batch, max_pair_num)

        final_pairwise_loss_example = torch.sum(pairwise_loss_clean,-1)/(pairs_num.float() + 1e-20)

        final_pairwise_loss = final_pairwise_loss_example.sum()/batch

        loss = new_original_loss + final_pairwise_loss * self.pairwise_loss_lam


        return loss

    def encode(self, input_ids, attention_mask=None, token_type_ids=None,
                pairs_list=None, passage_length=None, pairs_num=None, sep_positions=None, ground_truth=None, 
                mask_cls=None, pairwise_labels=None, cuda=None, head_mask=None):

        batch_size, max_pair_num, max_pair_len = input_ids.size()

        input_ids = input_ids.reshape(batch_size*max_pair_num, max_pair_len)
        attention_mask = attention_mask.reshape(batch_size*max_pair_num, max_pair_len)
        token_type_ids = token_type_ids.reshape(batch_size*max_pair_num, max_pair_len)


        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            head_mask=head_mask)

        top_vec = outputs[0]
        cls_pooled_output = outputs[1]


        final_seq_matriax, cls_output_matrix_nn, cls_score, cls_score_matrix_nn, cls_score_matrix_nn_his1, cls_score_matrix_nn_his2 = self.two_level_encoder(top_vec, pairs_list, passage_length, pairs_num, sep_positions, cuda, mask_cls, cls_pooled_output)

        clean_sents_vec = final_seq_matriax * mask_cls[:, :, None].float()



        mask_cls_tranf = mask_cls.to(dtype=next(self.parameters()).dtype)
        para_matrix = self.encoder(clean_sents_vec, mask_cls_tranf)

        clean_para_matrix = para_matrix * mask_cls[:, :, None].float()

        num_sen_newsize = (passage_length.float() + 1e-20).unsqueeze(1).expand(-1, self.hidden_size)

        para_vec = torch.sum(clean_para_matrix, 1)/num_sen_newsize



        hn = para_vec.unsqueeze(0)
        cn = torch.zeros_like(hn)
        hcn = (hn, cn)


        keyinput = torch.cat((clean_sents_vec, clean_para_matrix), -1)
        key = self.key_linear(keyinput)

        

        return clean_sents_vec, clean_para_matrix, hcn, key, cls_pooled_output, cls_output_matrix_nn, cls_score, cls_score_matrix_nn, cls_score_matrix_nn_his1, cls_score_matrix_nn_his2


    def step(self, prev_y, prev_handc, original_keys, mask, rela_vec, rela_mask, hist_left1, hist_left2, l1_mask, l2_mask):
        _, (h, c) = self.decoder(prev_y, prev_handc)

        query = h.squeeze(0).unsqueeze(1)
        query = self.query_linear(query)

        ## history ##
        left1 = hist_left1.masked_fill(l1_mask.unsqueeze(-1) == 0, 0).sum(1)
        left2 = hist_left2.masked_fill(l2_mask.unsqueeze(-1) == 0, 0).sum(1)


        # future
        rela_vec.masked_fill_(rela_mask.unsqueeze(-1) == 0, 0)
        forw_futu = rela_vec.mean(2)
        back_futu = rela_vec.mean(1)


        pw = torch.cat((left1, left2, forw_futu, back_futu), -1)
        keys = self.pw_k(pw)


        e = torch.tanh(query + keys + original_keys)

        e = self.tanh_linear(e).squeeze(2)

        e.masked_fill_(mask, -1e9)
        logp = F.log_softmax(e, dim=-1)

        return h, c, logp


def beam_search_pointer(args, model, input_ids, attention_mask=None, token_type_ids=None,
                pairs_list=None, passage_length=None, pairs_num=None, sep_positions=None, ground_truth=None, 
                mask_cls=None, pairwise_labels=None, cuda=None, head_mask=None):

    sentences, _, dec_init, original_keys, cls_pooled_output, cls_output_matrix_nn, cls_score, cls_score_matrix_nn, cls_score_matrix_nn_his1, cls_score_matrix_nn_his2 = model.encode(input_ids, attention_mask, token_type_ids, pairs_list, passage_length, pairs_num, 
            sep_positions, ground_truth, mask_cls, pairwise_labels, cuda, head_mask)

    num_sen = passage_length

    sentences = sentences[:,:num_sen,:]

    original_keys = original_keys[:,:num_sen,:]

    document = sentences.squeeze(0)


    T, H = document.size()

    rela_vec = model.rela_encode(cls_output_matrix_nn, cls_score_matrix_nn)

    hist_left1, hist_left2 = model.history_encode(cls_output_matrix_nn, cls_score_matrix_nn, cls_score_matrix_nn)


    eye_mask = torch.eye(T).cuda(int(cuda[-1])).byte()
    eye_zeros = torch.ones_like(eye_mask) - eye_mask

    W = args.beam_size

    prev_beam = Beam(W)
    prev_beam.candidates = [[]]
    prev_beam.scores = [0]

    target_t = T - 1

    f_done = (lambda x: len(x) == target_t)

    valid_size = W
    hyp_list = []

    for t in range(target_t):
        candidates = prev_beam.candidates
        if t == 0:
            # start
            dec_input = sentences.new_zeros(1, 1, H)
            pointed_mask = sentences.new_zeros(1, T).byte()

            rela_mask = eye_zeros.unsqueeze(0).cuda(int(cuda[-1]))

            l1_mask = torch.zeros_like(rela_mask).cuda(int(cuda[-1]))
            l2_mask = torch.zeros_like(rela_mask).cuda(int(cuda[-1]))

        else:
            index = sentences.new_tensor(list(map(lambda cand: cand[-1], candidates))).long()

            dec_input = document[index].unsqueeze(1)

            temp_batch = index.size(0)

            pointed_mask[torch.arange(temp_batch), index] = 1

            rela_mask[torch.arange(temp_batch), :, index] = 0
            rela_mask[torch.arange(temp_batch), index] = 0

            l1_mask = torch.zeros_like(rela_mask).cuda(int(cuda[-1]))
            l2_mask = torch.zeros_like(rela_mask).cuda(int(cuda[-1]))

            l1_mask[torch.arange(temp_batch), index, :] = 1 

            if t > 1:
                left2_index = index.new_tensor(list(map(lambda cand: cand[-2], candidates)))
                l2_mask[torch.arange(temp_batch), left2_index, :] = 1


        dec_h, dec_c, log_prob = model.step(dec_input, dec_init, original_keys, pointed_mask, rela_vec, rela_mask, 
            hist_left1, hist_left2, l1_mask, l2_mask)

        next_beam = Beam(valid_size)
        done_list, remain_list = next_beam.step(-log_prob, prev_beam, f_done)
        hyp_list.extend(done_list)
        valid_size -= len(done_list)

        if valid_size == 0:
            break


        beam_remain_ix = input_ids.new_tensor(remain_list)

        dec_h = dec_h.index_select(1, beam_remain_ix)
        dec_c = dec_c.index_select(1, beam_remain_ix)
        dec_init = (dec_h, dec_c)

        pointed_mask = pointed_mask.index_select(0, beam_remain_ix)

        rela_mask = rela_mask.index_select(0, beam_remain_ix)
        rela_vec = rela_vec.index_select(0, beam_remain_ix)

        hist_left1 = hist_left1.index_select(0, beam_remain_ix)
        hist_left2 = hist_left2.index_select(0, beam_remain_ix)

        prev_beam = next_beam


    score = dec_h.new_tensor([hyp[1] for hyp in hyp_list])
    sort_score, sort_ix = torch.sort(score)
    output = []
    for ix in sort_ix.tolist():
        output.append((hyp_list[ix][0], score[ix].item()))
    best_output = output[0][0]

    the_last = list(set(list(range(T))).difference(set(best_output)))
    best_output.append(the_last[0])

    return best_output



















