import json
import tensorflow as tf
import numpy as np
from collections import OrderedDict
from collections import Counter


# 3-class classification demo
class Classifier(object):
    def __init__(self, corpus_path, min_freq=1, special=['<PAD>', '<UNKNOWN>']):
        assert corpus_path, "corpus_path must be given."
        self.max_seq_len = 6
        self.batch_size = 4
        self.iter_num = 50
        self.min_freq = min_freq
        self.special = special
        self.counter = Counter()
        self.vocab = OrderedDict()
        self.corpus = self.load_corpus(corpus_path)
        self.feed_dict = {}

    def load_corpus(self, corpus_path):
        ret_corpus = OrderedDict()
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                text, label = data["text"], data['items'][0]['sentiment']
                self.counter.update(list(text))
                ret_corpus[text] = int(label)
        return ret_corpus

    def build_vocab(self):
        if self.special:
            for token in self.special:
                self.vocab[token] = len(self.vocab)
        # select 20 words in corpus
        for token, cnt in self.counter.most_common(20):
            if cnt < self.min_freq:
                break
            self.vocab[token] = len(self.vocab)

    def build_embedding(self, word_ids):
        # first token is padding, set it untrainable
        embed_t = tf.get_variable(shape=[1, 8], trainable=False, name="embed_t",
                                  initializer=tf.zeros_initializer())
        embed_ut = tf.get_variable(shape=[len(self.vocab) - 1, 8], trainable=True, name="embed_ut",
                                   initializer=tf.truncated_normal_initializer())
        embedding = tf.concat([embed_t, embed_ut], axis=0, name="embedd_w")
        return tf.nn.embedding_lookup(embedding, word_ids)

    def models(self, word_ids):
        embeded_sent = self.build_embedding(word_ids)
        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=8)
        initial_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(lstm_cell, embeded_sent, initial_state=
                                           initial_state)
        # full connected layer
        logits = tf.layers.dense(state[1], units=3)
        return logits

    def train(self):
        global_steps = tf.Variable(0, trainable=False, name="global_steps")
        word_ids = tf.placeholder(tf.int32, [None, self.max_seq_len], name="input_text")
        label = tf.placeholder(tf.int32, [None], name="label")
        logits = self.models(word_ids)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        optimizer = tf.train.AdamOptimizer(2.5e-4)
        grad_and_var = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grad_and_var, global_step=global_steps)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # generator for training process
            train_gens = self.get_batch(self.batch_size)
            for iter in range(self.iter_num):
                texts, labels = next(train_gens)
                _, iter_num, loss_eval, embed_w = sess.run([train_op, global_steps, loss, "embedd_w:0"],
                                                           feed_dict={word_ids: texts, label: labels})
                print("-" * 30 + "iteration {}".format(iter_num) + "-" * 30)
                print(embed_w[0:2, :])

    def get_batch(self, batch_size=4):
        train_data = list(self.corpus.items())

        def text2id(query):
            ret = []
            for i in range(self.max_seq_len):
                if i < len(query):
                    ret.append(self.vocab[query[i]] if query[i] in self.vocab
                               else self.vocab["<UNKNOWN>"])
                else:
                    ret.append(self.vocab["<PAD>"])
            return ret
        # get a batch data
        while True:
            batch = np.random.choice(len(train_data), batch_size, replace=False)
            batch = [train_data[idx] for idx in batch]
            texts, labels = [], []
            for text, lbl in batch:
                # transform word to id
                texts.append(text2id(text))
                labels.append(lbl)
            yield texts, labels


if __name__ == "__main__":
    classifier = Classifier("./result.txt")
    classifier.build_vocab()
    classifier.train()
