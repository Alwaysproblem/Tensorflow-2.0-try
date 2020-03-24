
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import defaultdict
import pandas as pd
# from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter


class Dataset(object):
    def __init__(self, file_dir):
        data = []
        with open(file_dir, 'r') as f:
            for i, row in enumerate(f.readlines()):
                if i == 0: continue
                u, i, _, _ = row.split(',')
                data.append((int(u), int(i)))
        self.train_data = np.asarray(data[:int(len(data) * 0.7)])
        self.test_data = np.asarray(data[int(len(data) * 0.7):])
        self._train_index = np.arange(len(self.train_data), dtype=np.uint)
        self.users_nb, self.items_nb = self.train_data.max(axis=0) + 1

        # Neighborhoods
        self.user_items = defaultdict(set)
        self.item_users = defaultdict(set)
        self.item_users_list = defaultdict(list)
        for u, i in self.train_data:
            self.user_items[u].add(i)
            self.item_users[i].add(u)
            self.item_users_list[i].append(u)
        self._max_user_neighbors = max(len(x) for x in self.item_users_list.values())

    def _sample_negative_item(self, user_id):
        n = np.random.randint(0, self.items_nb)
        positive_items = self.user_items[user_id]
        #         print(type(positive_items))
        while n in positive_items or n not in self.item_users:
            n = np.random.randint(0, self.items_nb)
        return n

    def get_data(self, batch_size, neighborhood, neg_count):
        batch = np.zeros((batch_size, 3), dtype=np.uint32)
        #         print(self._max_user_neighbors)
        pos_neighbor = np.zeros((batch_size, self._max_user_neighbors), dtype=np.uint32)
        pos_length = np.zeros(batch_size, dtype=np.int32)
        neg_neighbor = np.zeros((batch_size, self._max_user_neighbors), dtype=np.int32)
        neg_length = np.zeros(batch_size, dtype=np.int32)
        # shuffle index
        np.random.shuffle(self._train_index)

        idx = 0
        for user_idx, item_idx in self.train_data[self._train_index]:
            # TODO: set positive values outside of for loop
            for _ in range(neg_count):
                neg_item_idx = self._sample_negative_item(user_idx)
                batch[idx, :] = [user_idx, item_idx, neg_item_idx]
                if neighborhood:
                    if len(self.item_users[item_idx]) > 0:
                        pos_length[idx] = len(self.item_users[item_idx])
                        pos_neighbor[idx, :pos_length[idx]] = self.item_users_list[item_idx]
                    else:
                        pos_length[idx] = 1
                        pos_neighbor[idx, 0] = item_idx
                    if len(self.item_users[neg_item_idx]) > 0:
                        neg_length[idx] = len(self.item_users[neg_item_idx])
                        neg_neighbor[idx, :neg_length[idx]] = self.item_users_list[neg_item_idx]
                    else:
                        # Length defaults to 1
                        neg_length[idx] = 1
                        neg_neighbor[idx, 0] = neg_item_idx
                idx += 1
                if idx == batch_size:
                    if neighborhood:
                        max_length = max(neg_length.max(), pos_length.max())
                        yield batch, pos_neighbor[:, :max_length], pos_length, \
                              neg_neighbor[:, :max_length], neg_length
                        pos_length[:] = 1
                        neg_length[:] = 1
                    else:
                        yield batch
                    idx = 0
        if idx > 0:
            if neighborhood:
                max_length = max(neg_length[:idx].max(), pos_length[:idx].max())
                yield batch[:idx], pos_neighbor[:idx, :max_length], pos_length[:idx], \
                      neg_neighbor[:idx, :max_length], neg_length[:idx]
            else:
                yield batch[:idx]


def ApplyAttentionMemory(memory, output_memory, query, memory_mask=None, maxlen=None):
    # query = [batch size, embeddings] => Expand => [batch size, embeddings, 1]
    #         Transpose => [batch size, 1, embeddings]
    #     query = tf.expand_dims(query,1)
    query_expanded = query * memory

    # Return: [batch size, max length]
    scores = tf.reduce_sum(query_expanded, axis=2)
    attention = keras.activations.softmax(scores)
    probs_temp = tf.expand_dims(attention, 1)
    c_temp = tf.transpose(output_memory, [0, 2, 1])

    #     attention = tf.expand_dims(attention,1)
    #     print(attention)
    #     print(output_memory)

    neighborhood = probs_temp * c_temp
    weighted_output = tf.reduce_sum(neighborhood, axis=2)
    return attention, weighted_output


class CollaborativeMemoryNetwork(tf.keras.Model):
    def __init__(self, config):
        super(CollaborativeMemoryNetwork, self).__init__(name='cmn')
        # inputs
        self.config = config
        self._initializer()
        # self.optimizer = keras.optimizers.RMSprop(self.config.lr)

    def _initializer(self):
        self.embed_init = keras.initializers.TruncatedNormal(stddev=0.01)
        self._init = keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal')
        self._output_init = keras.initializers.glorot_uniform()
        # embedding layer
        self.user_memory = keras.layers.Embedding(self.config.user_count,
                                                  self.config.embedding_size,
                                                  embeddings_initializer=self.embed_init)
        self.user_output = keras.layers.Embedding(self.config.user_count,
                                                  self.config.embedding_size,
                                                  embeddings_initializer=self.embed_init)
        self._output_Dense = keras.layers.Dense(self.config.embedding_size,
                                                use_bias=True,
                                                kernel_initializer=self._init,
                                                kernel_regularizer=keras.regularizers.l2(self.config.l2))
        self._output_1 = keras.layers.Dense(1,
                                        use_bias=False,
                                        activation='relu',
                                        kernel_initializer=self._output_init,
                                        kernel_regularizer=keras.regularizers.l2(self.config.l2))
        self.item_memory = keras.layers.Embedding(self.config.item_count,
                                                  self.config.embedding_size,
                                                  embeddings_initializer=self.embed_init)
        self.Dense_layers = [keras.layers.Dense(self.config.embedding_size,
                                                kernel_initializer=self._init,
                                                use_bias=True,
                                                bias_initializer='ones',
                                                kernel_regularizer=keras.regularizers.l2(self.config.l2),
                                                ) for i in range(self.config.hops - 1)]

    def call(self, inputs):
        self.input_users, self.input_items, self.input_items_negative, \
        self.input_neighborhoods, self.input_neighborhood_lengths, \
        self.input_neighborhoods_negative, self.input_neighborhood_lengths_negative = \
            (tf.convert_to_tensor(e) for e in inputs)
        self._cur_user = self.user_memory(self.input_users)
        self._cur_user_output = self.user_output(self.input_users)
        # items memories a query
        self._cur_item = self.item_memory(self.input_items)
        self._cur_item_negative = self.item_memory(self.input_items_negative)
        self._cur_neighbors_negative_memory = self.user_memory(self.input_neighborhoods_negative)
        self._cur_neighbors_negative_output = self.user_output(self.input_neighborhoods_negative)
        self._cur_neighbors_memory = self.user_memory(self.input_neighborhoods)
        self._cur_neighbors_output = self.user_output(self.input_neighborhoods)

        # share Embeddings

        self._cur_item_output = self._cur_item
        self._cur_item_output_negative = self._cur_item_negative

        # our query = m_u + e_i
        self.query = (self._cur_user, self._cur_item)
        self.neg_query = (self._cur_user, self._cur_item_negative)
        self.neighbor = self.hop_layer(self.query,
                                       self._cur_neighbors_memory,
                                       self._cur_neighbors_output,
                                       self.config.max_neighbors)[-1][-1]
        aa = tf.multiply(self._cur_user, self._cur_item)
        aa = tf.reshape(aa, [-1, self.config.embedding_size])
        yian = tf.concat([aa, self.neighbor], axis=-1)
        self.score = self._output_1(self._output_Dense(yian))
    def hop_layer(self, query, memory, output_memory, seq_length, maxlen=32):
        user_query, item_query = query
        hop_outputs = []

        query = user_query + item_query
        #         print(query)
        memory_hop = [0., 0.]
        for hop_k in range(self.config.hops):
            if hop_k > 0:
                query = self.Dense_layers[hop_k - 1](query)
                query = keras.activations.relu(query + tf.expand_dims(memory_hop[-1], 1))

            memory_hop = ApplyAttentionMemory(memory, output_memory, query, seq_length, maxlen=maxlen)
            hop_outputs.append(memory_hop)
        return hop_outputs
    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        inputs, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            self.call(inputs)
            neighbor_negative = self.hop_layer(self.query,
                                               self._cur_neighbors_negative_memory,
                                               self._cur_neighbors_negative_output,
                                               self.config.max_neighbors)[-1][-1]
            aa = tf.multiply(self._cur_user, self._cur_item_negative)
            aa = tf.reshape(aa, [-1, self.config.embedding_size])
            yian = tf.concat([aa, neighbor_negative], axis=-1)
            negative_output = self._output_1(self._output_Dense(yian))
            y_pred = self.score - negative_output
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)
            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}
    # @tf.function
    # def train(self, input_users, input_items, input_items_negative, pos_neighborhoods, pos_neighborhood_length,
    #           neg_neighborhoods, neg_neighborhood_length):
    #     # Negative
    #     with tf.GradientTape() as tape:
    #         self.call([input_users, input_items, input_items_negative, pos_neighborhoods, pos_neighborhood_length,
    #                    neg_neighborhoods, neg_neighborhood_length])
    #         neighbor_negative = self.hop_layer(self.query,
    #                                            self._cur_neighbors_negative_memory,
    #                                            self._cur_neighbors_negative_output,
    #                                            self.config.max_neighbors)[-1][-1]
    #         aa = tf.multiply(self._cur_user, self._cur_item_negative)
    #         aa = tf.reshape(aa, [-1, self.config.embedding_size])
    #         yian = tf.concat([aa, neighbor_negative], axis=-1)
    #         negative_output = self._output_1(self._output_Dense(yian))
    #         eps = 1e-12
    #         loss = tf.reduce_mean(-tf.math.log(keras.activations.sigmoid((self.score - negative_output) + eps)))
    #         grads = tape.gradient(loss, self.trainable_variables)
    #         self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
    #     return loss


class Config(object):
    def __init__(self):
        self.iters = 5 #Max iters
        self.batch_size = 128
        self.embedding_size = 50
        self.data_file = './data/ml-latest-small/ratings.csv'
        self.hops = 2
        self.neg_count = 4 #Negative Samples Count
        self.l2 = 0.1 #l2 Regularization
        self.user_count = -1
        self.item_count = -1
        self.tol = 1e-5
        self.grad_clip = 5.0
        self.decay_rate = 0.9
        self.lr = 1e-4
        max_neighbors = -1
config = Config()

dataset = Dataset(config.data_file)
config.item_count = dataset.items_nb
config.user_count = dataset.users_nb
config.max_neighbors = dataset._max_user_neighbors

model = CollaborativeMemoryNetwork(config)
def loss_fn(y_true,y_pre):
    return tf.reduce_mean(-tf.math.log(keras.activations.sigmoid((y_pre) + 1e-12)))
model.compile(optimizer=keras.optimizers.RMSprop(config.lr),
              loss = loss_fn)
# tf.config.experimental_run_functions_eagerly(True)
for i in range(config.iters):
    progress = dataset.get_data(config.batch_size,True,config.neg_count)
    for example in progress:
        ratings,pos_neighborhoods,pos_neighborhood_length,neg_neighborhoods,neg_neighborhood_length = example
        batch_s = ratings.shape[0]
        data_set = tf.data.Dataset.from_tensor_slices((ratings,pos_neighborhoods,pos_neighborhood_length,neg_neighborhoods,neg_neighborhood_length))
        input_users = np.expand_dims(ratings[:,0],-1)
        input_items = np.expand_dims(ratings[:,1],-1)
        input_items_negative = np.expand_dims(ratings[:,2],-1)
        total_inputs = [input_users,input_items,input_items_negative,pos_neighborhoods,pos_neighborhood_length,neg_neighborhoods,neg_neighborhood_length]
        y_trues = np.ones([batch_s,1])
        history = model.fit(total_inputs,y_trues,verbose=True)
        # print(loss.numpy())
tf.config.experimental_run_functions_eagerly(False)






