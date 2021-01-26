from .base_graph import BaseGraph
from robonet.video_prediction.layers.dnaflow_rnn_cell import VPredCell
import itertools
import tensorflow as tf
from robonet.video_prediction.utils import tf_utils


class DNAFlowGraphWrapper(BaseGraph):
    def build_graph(self, mode, inputs, hparams, n_gpus=1, scope_name='dnaflow_generator'):
        if hparams.use_states:
            assert "states" in inputs, "graph is building with states but no states in inptus"
        else:
            inputs.pop('states', None)
        
        self._scope_name = scope_name
        outputs_enc = inputs.pop('outputs_enc', None)
        with tf.variable_scope(self._scope_name) as graph_scope:
            # TODO: I really don't like this. Should just error at this point instead of padding
            inputs = {name: tf_utils.maybe_pad_or_slice(input, hparams.sequence_length - 1)
                for name, input in inputs.items()}

            if outputs_enc is not None:
                inputs['e'] = outputs_enc

            cell = VPredCell(mode, inputs, hparams)
            outputs, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs['actions'].dtype,
                                    swap_memory=False, time_major=True)
        
            outputs = {name: output[hparams.context_frames - 1:] for name, output in outputs.items()}
            outputs['ground_truth_sampling_mean'] = tf.reduce_mean(tf.to_float(cell.ground_truth[hparams.context_frames:]))
        return outputs

    @staticmethod
    def default_hparams():
        default_params =  {
            "where_add": "all",
            'last_frames': 2,
            'num_transformed_images': 4,
            'prev_image_background': True,
            'first_image_background': True,
            'context_images_background': False,
            'generate_scratch_image': False,
            'transformation': "flow",
            'conv_rnn': "lstm",
            'norm_layer': "instance",
            'ablation_conv_rnn_norm': False,
            'downsample_layer': "conv_pool2d",
            'upsample_layer': "upsample_conv2d",
            'dependent_mask': True,
            'c_dna_kernel_size': [5, 5],              # only used in CDNA/DNA mode

            'schedule_sampling': "inverse_sigmoid",
            'schedule_sampling_k': 900.0,
            'schedule_sampling_steps': [0, 100000],
            
            'renormalize_pixdistrib': True,

            # params below control size of model
            'ngf': 32,
            'encoder_layer_size_mult': [1, 2, 4],
            'encoder_layer_use_rnn': [True, True, True],
            'decoder_layer_size_mult': [2, 1, 1],
            'decoder_layer_use_rnn': [True, True, False]
        }
        return dict(itertools.chain(BaseGraph.default_hparams().items(), default_params.items()))
