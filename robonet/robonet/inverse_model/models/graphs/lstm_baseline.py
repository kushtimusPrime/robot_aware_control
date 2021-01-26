from robonet.inverse_model.models.graphs.base_graph import BaseGraph
import itertools
import tensorflow as tf
import tensorflow.keras.layers as layers
from robonet.inverse_model.models.layers.vgg_pretrain import get_vgg_dict, vgg_preprocess_images, vgg_conv, vgg_pool


class ImageEncoder(tf.Module):
    def __init__(self, conv_filters, kernel_size, out_dim, vgg_path, n_convs=3, padding='same', fc_layer=256):
        self._vgg_dict = get_vgg_dict(vgg_path)
        
        self._convs = [[layers.Conv2D(conv_filters, kernel_size, padding="same", dilation_rate=min(c + 1, 3)), 
                        layers.BatchNormalization(axis=-1)] for c in range(n_convs)]

        # top layer
        self._fc_layer = [layers.Dense(fc_layer), layers.BatchNormalization(axis=-1)]
        self._top = [layers.Dense(out_dim), layers.BatchNormalization(axis=-1)]
 
    def __call__(self, input_img, training=True):
        preprocessed = vgg_preprocess_images(input_img)
        conv1_out = vgg_conv(self._vgg_dict, vgg_conv(self._vgg_dict, preprocessed, "conv1_1"), "conv1_2")
        conv1_out = vgg_pool(conv1_out, "pool1")

        conv2_out = vgg_conv(self._vgg_dict, vgg_conv(self._vgg_dict, conv1_out, "conv2_1"), "conv2_2")
        conv2_out = vgg_pool(conv2_out, "pool2")

        conv3_out = conv2_out
        for c in ['conv3_1', 'conv3_2', 'conv3_3', 'conv3_4']:
            conv3_out = vgg_conv(self._vgg_dict, conv3_out, c)
        conv3_out = vgg_pool(conv3_out, "pool3")

        conv4_out = conv3_out
        for c in ['conv4_1', 'conv4_2', 'conv4_3', 'conv4_4']:
            conv4_out = vgg_conv(self._vgg_dict, conv4_out, c)
        conv4_out = vgg_pool(conv4_out, "pool4")

        top = vgg_conv(self._vgg_dict, conv4_out, "conv5_1")
        for layer in self._convs:
            conv, norm = layer
            top = norm(tf.nn.relu(conv(top))) + top
        
        top = tf.nn.max_pool(top, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='top_pool')
        dense, norm = self._fc_layer
        top = norm(tf.nn.relu(dense(tf.reshape(top, (top.get_shape().as_list()[0], -1)))))
        
        dense, norm = self._top
        return norm(tf.nn.relu(dense(top)))
 

class LSTMBaseline(BaseGraph):
    def build_graph(self, mode, inputs, hparams, scope_name='flow_generator'):
        is_train = mode == tf.estimator.ModeKeys.TRAIN
        B = inputs['start_images'].get_shape().as_list()[0]
        self._scope_name = scope_name
        outputs = {}
        with tf.variable_scope(scope_name):
            encoder = ImageEncoder(hparams.conv_filters, hparams.kernel_size, hparams.enc_dim, hparams.vgg_path, hparams.n_convs)
            start_enc = encoder(inputs['start_images'], training=is_train)
            goal_enc = encoder(inputs['goal_images'], training=is_train)
            start_goal_enc = tf.concat((start_enc, goal_enc), -1)

            lstm_in = layers.Dense(hparams.latent_dim * inputs['T'])(start_goal_enc)
            lstm_in = layers.BatchNormalization(axis=-1)(tf.nn.relu(lstm_in), training=is_train)
            lstm_in = tf.reshape(lstm_in, (-1, inputs['T'], hparams.latent_dim))
            
            lstm_dim = hparams.latent_dim 
            if hparams.append_last_action:
                lstm_dim += + inputs['adim']
            
            lstm = layers.LSTM(lstm_dim)
            lstm.cell.build([B, inputs['T'], lstm_dim])

            if 'context_actions' in inputs:
                last_action = inputs['context_actions'][:, -1]
            else:
                last_action = tf.zeros((B, inputs['adim']))

            action_predictions = []
            top_layer = layers.Dense(inputs['adim'])
            schedule_sample = self.schedule_sample(inputs['T'], B, hparams)
            for t in range(inputs['T']):
                if hparams.append_last_action:
                    if t > 0 and is_train:
                        real_action = inputs['real_actions'][:, t - 1]
                        last_action = tf.where(schedule_sample[t - 1], real_action, action_predictions[-1][:, 0])
                    elif t > 0:
                        last_action = action_predictions[-1][:, 0]
                    in_t = tf.concat([lstm_in[:, t], last_action], axis=-1)
                else:
                    in_t = lstm_in[:, t]

                if t == 0:
                    if 'context_frames' in inputs:
                        assert hparams.append_last_action
                        context_encodings = [encoder(inputs['context_frames'][:, c]) for c in range(hparams.context_actions)]

                        for i, c in enumerate(context_encodings):
                            dense = layers.Dense(hparams.latent_dim)(c)
                            context_enc = layers.BatchNormalization(axis=-1)(tf.nn.relu(dense), training=is_train)
                            if i == 0:
                                context_act = tf.zeros_like(inputs['context_actions'][:, 0])
                            else:
                                context_act = inputs['context_actions'][:, i - 1]
                            
                            context_in = tf.concat((context_enc, context_act), axis=-1)
                            if i == 0:
                                hidden_state = lstm.get_initial_state(context_in[:, None])
                            _, hidden_state = lstm.cell(context_in, hidden_state)
                    else:
                        hidden_state = lstm.get_initial_state(in_t[:, None])

                lstm_out, hidden_state = lstm.cell(in_t, hidden_state)
                action_predictions.append(top_layer(lstm_out)[:, None])
                

            outputs['pred_actions'] = tf.concat(action_predictions, axis=1)
            if hparams.append_last_action and inputs['T'] > 1:
                outputs['ground_truth_sampling_mean'] = tf.reduce_mean(tf.to_float(schedule_sample))

            return outputs

    @staticmethod
    def default_hparams():
        default_params =  {
            "n_convs": 3,
            "conv_filters": 512,
            "enc_dim": 128,
            "kernel_size": 3,

            "latent_dim": 20,

            "vgg_path": '~/',
            "append_last_action": True,
            "schedule_sampling_k": 900.0,
            "schedule_sampling_steps": [0, 100000],
        }
        return dict(itertools.chain(BaseGraph.default_hparams().items(), default_params.items()))

    def schedule_sample(self, T, B, hparams):
        if T == 1:
            return

        ground_truth_sampling_shape = [T - 1, B]
        
        k = hparams.schedule_sampling_k
        start_step = hparams.schedule_sampling_steps[0]
        iter_num = tf.to_float(tf.train.get_or_create_global_step())
        prob = (k / (k + tf.exp((iter_num - start_step) / k)))
        prob = tf.cond(tf.less(iter_num, start_step), lambda: 1.0, lambda: prob)

        log_probs = tf.log([1 - prob, prob])
        ground_truth_sampling = tf.multinomial([log_probs] * B, ground_truth_sampling_shape[0])
        ground_truth_sampling = tf.cast(tf.transpose(ground_truth_sampling, [1, 0]), dtype=tf.bool)
        # Ensure that eventually, the model is deterministically
        # autoregressive (as opposed to autoregressive with very high probability).
        ground_truth_sampling = tf.cond(tf.less(prob, 0.001),
                                        lambda: tf.constant(False, dtype=tf.bool, shape=ground_truth_sampling_shape),
                                        lambda: ground_truth_sampling)
        return ground_truth_sampling