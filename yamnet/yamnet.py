# Copyright 2019 The TensorFlow Authors All Rights Reserved.
# ... (license headers) ...

"""Core model definition of YAMNet."""

import csv
import numpy as np
import tensorflow as tf
import warnings
from tensorflow.keras import Model, layers # Use tensorflow.keras

import features as features_lib # Should define YamnetFeaturesLayer
import params as yamnet_params_module

warnings.filterwarnings("ignore", message="Gradients do not exist for variables.*when minimizing the loss")

# --- Layer Helper Functions ---
def _batch_norm(name, params):
    keras_compatible_name = name.replace('/', '_')
    def _bn_layer(layer_input):
        return layers.BatchNormalization(
            name=keras_compatible_name, center=params.batchnorm_center,
            scale=params.batchnorm_scale, epsilon=params.batchnorm_epsilon)(layer_input)
    return _bn_layer

def _conv(name, kernel, stride, filters, params):
    keras_compatible_name = name.replace('/', '_') # For by_name=True robustness if needed
    def _conv_layer(layer_input):
        output = layers.Conv2D(name=f'{keras_compatible_name}_conv', filters=filters, kernel_size=kernel, strides=stride,
                               padding=params.conv_padding, use_bias=False, activation=None)(layer_input)
        output = _batch_norm(f'{keras_compatible_name}_conv_bn', params)(output)
        output = layers.ReLU(name=f'{keras_compatible_name}_relu')(output)
        return output
    return _conv_layer

def _separable_conv(name, kernel, stride, filters, params):
    keras_compatible_name = name.replace('/', '_') # For by_name=True robustness
    def _separable_conv_layer(layer_input):
        output = layers.DepthwiseConv2D(name=f'{keras_compatible_name}_depthwise_conv', kernel_size=kernel, strides=stride,
                                        depth_multiplier=1, padding=params.conv_padding, use_bias=False,
                                        activation=None)(layer_input)
        output = _batch_norm(f'{keras_compatible_name}_depthwise_conv_bn', params)(output)
        output = layers.ReLU(name=f'{keras_compatible_name}_depthwise_conv_relu')(output)
        output = layers.Conv2D(name=f'{keras_compatible_name}_pointwise_conv', filters=filters, kernel_size=(1, 1), strides=1,
                               padding=params.conv_padding, use_bias=False, activation=None)(output)
        output = _batch_norm(f'{keras_compatible_name}_pointwise_conv_bn', params)(output)
        output = layers.ReLU(name=f'{keras_compatible_name}_pointwise_conv_relu')(output)
        return output
    return _separable_conv_layer

_YAMNET_LAYER_DEFS = [
    (_conv,          [3, 3], 2,   32), (_separable_conv, [3, 3], 1,   64),
    (_separable_conv, [3, 3], 2,  128), (_separable_conv, [3, 3], 1,  128),
    (_separable_conv, [3, 3], 2,  256), (_separable_conv, [3, 3], 1,  256),
    (_separable_conv, [3, 3], 2,  512), (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512), (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 1,  512), (_separable_conv, [3, 3], 1,  512),
    (_separable_conv, [3, 3], 2, 1024), (_separable_conv, [3, 3], 1, 1024)
]

def yamnet_frames_model(params):
    """Defines the YAMNet waveform-to-class-scores model using YamnetFeaturesLayer.
       MODIFIED to output embeddings and log_mel_spectrogram for transfer learning.
    """
    waveform = layers.Input(shape=(None,), dtype=tf.float32, name='waveform_input')

    features_layer = features_lib.YamnetFeaturesLayer(params=params, name="yamnet_features") # Added name
    log_mel_spectrogram, features = features_layer(waveform)
    # features shape: (batch_size, num_patches, patch_frames, mel_bands)

    # --- Reshape features to (batch*patches, frames, bands) ---
    # This section seems to be from the version 1 you posted.
    def calculate_reshape_target_for_features(feat_tensor):
        s = tf.shape(feat_tensor)
        # Target: [batch * patches, frames, bands]
        # Ensure params.patch_frames and params.patch_bands are accessible
        return tf.stack([s[0] * s[1], params.patch_frames, params.patch_bands])

    # Layer to calculate the target shape dynamically
    target_shape_tensor_for_features = layers.Lambda(
        calculate_reshape_target_for_features,
        name='calc_feat_reshape_shape' # Name from your successful summary
    )(features)

    # Layer to perform the reshape using the dynamically calculated shape
    def reshape_features_lambda_op(args):
        feat_tensor, target_s = args
        return tf.reshape(feat_tensor, target_s)

    features_reshaped_output_shape = (params.patch_frames, params.patch_bands) # Output shape for one item
    features_reshaped = layers.Lambda(
        reshape_features_lambda_op,
        output_shape=features_reshaped_output_shape, # CRITICAL: Added output_shape
        name='features_reshape' # Name from your successful summary
    )([features, target_shape_tensor_for_features])
    # Shape now: (batch*patches, patch_frames, patch_bands)

    # --- Add channel dimension ---
    net = layers.Reshape(
        (params.patch_frames, params.patch_bands, 1), name='input_reshape_channel' # Name from summary
    )(features_reshaped)
    # Shape now: (batch*patches, patch_frames, patch_bands, 1)

    # --- Apply CNN stack ---
    for (i, (layer_fun, kernel, stride, filters)) in enumerate(_YAMNET_LAYER_DEFS):
        # Use Keras compatible names for layers if strict loading fails,
        # otherwise try to match original naming from yamnet.h5 (often with '/')
        # The successful summary used 'layer1_conv' etc., implying '_' was used.
        layer_name = f'layer{i + 1}'
        net = layer_fun(layer_name, kernel, stride, filters, params)(net)

    # --- Original Output layers (needed for weight loading) ---
    embeddings_per_patch = layers.GlobalAveragePooling2D(name='avg_pool')(net) # Name from summary
    # Define original logits and predictions to match the structure of yamnet.h5
    original_logits = layers.Dense(units=params.num_classes, use_bias=True, name='logits')(embeddings_per_patch) # Name from summary
    original_predictions_per_patch = layers.Activation(
        activation=params.classifier_activation, name='predictions_activation' # Name from summary
        )(original_logits)

    # --- Reshape original predictions and embeddings back to include patch dimension ---
    # This is also from your version 1, needed for consistency if weights depend on these layers existing.
    def reshape_outputs_lambda_op(tensors_list):
        flat_tensor, feat_tensor_for_shape = tensors_list # Use 'features' for shape reference
        batch_size = tf.shape(feat_tensor_for_shape)[0]
        num_patches = tf.shape(feat_tensor_for_shape)[1]
        last_dim = tf.shape(flat_tensor)[-1]
        return tf.reshape(flat_tensor, (batch_size, num_patches, last_dim))

    predictions_output_shape = (None, params.num_classes) # (num_patches, num_classes)
    _ignored_original_predictions = layers.Lambda( # We will ignore this output
        reshape_outputs_lambda_op,
        output_shape=predictions_output_shape,
        name='predictions_reshape' # Name from summary
    )([original_predictions_per_patch, features])

    embeddings_output_shape = (None, 1024) # (num_patches, embedding_size)
    final_embeddings = layers.Lambda( # This is the embedding we want
        reshape_outputs_lambda_op,
        output_shape=embeddings_output_shape,
        name='embeddings_reshape' # Name from summary
    )([embeddings_per_patch, features])

    # --- MODIFIED OUTPUT FOR FINE-TUNING ---
    # The successful script's BirdClassifier expects three outputs and takes the second (embeddings)
    # So, we should return all three that the original model (this version) produces.
    # Your BirdClassifier.call will then pick the embeddings.
    frames_model = Model(
        name='yamnet_frames', inputs=waveform,
        outputs=[_ignored_original_predictions, final_embeddings, log_mel_spectrogram])
    return frames_model

def class_names(class_map_csv):
    # (Keep this function as is from the version 1 or official version)
    if tf.is_tensor(class_map_csv): class_map_csv = class_map_csv.numpy()
    with open(class_map_csv) as csv_file:
        reader = csv.reader(csv_file); next(reader)
        return np.array([display_name for (_, _, display_name) in reader])