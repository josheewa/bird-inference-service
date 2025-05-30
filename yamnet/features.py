# Copyright 2019 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================

"""Feature computation for YAMNet, refactored into a Keras Layer."""

import numpy as np
import tensorflow as tf
# Use tf.keras directly
from tensorflow.keras import layers

# --- Keep helper functions if needed by the layer ---
# (e.g., _tflite_stft_magnitude - keep it for now)
def _tflite_stft_magnitude(signal, frame_length, frame_step, fft_length):
    # ... (Keep the implementation of _tflite_stft_magnitude as it was) ...
    def _hann_window():
      return tf.reshape(
        tf.constant(
            (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(0, 1.0, 1.0 / frame_length))
            ).astype(np.float32),
            name='hann_window'), [1, frame_length])

    def _dft_matrix(dft_length):
      """Calculate the full DFT matrix in NumPy."""
      omega = (0 + 1j) * 2.0 * np.pi / float(dft_length)
      return np.exp(omega * np.outer(np.arange(dft_length), np.arange(dft_length)))

    def _rdft(framed_signal, fft_length):
      """Implement real-input Discrete Fourier Transform by matmul."""
      complex_dft_matrix_kept_values = _dft_matrix(fft_length)[:(
          fft_length // 2 + 1), :].transpose()
      real_dft_matrix = tf.constant(
          np.real(complex_dft_matrix_kept_values).astype(np.float32),
          name='real_dft_matrix')
      imag_dft_matrix = tf.constant(
          np.imag(complex_dft_matrix_kept_values).astype(np.float32),
          name='imaginary_dft_matrix')
      signal_frame_length = tf.shape(framed_signal)[-1]
      half_pad = (fft_length - signal_frame_length) // 2
      padded_frames = tf.pad(
          framed_signal,
          [
              [0, 0], # Batch dimension
              [half_pad, fft_length - signal_frame_length - half_pad] # Frame dimension
          ],
          mode='CONSTANT',
          constant_values=0.0)
      real_stft = tf.matmul(padded_frames, real_dft_matrix)
      imag_stft = tf.matmul(padded_frames, imag_dft_matrix)
      return real_stft, imag_stft

    def _complex_abs(real, imag):
      return tf.sqrt(tf.add(real * real, imag * imag))

    framed_signal = tf.signal.frame(signal, frame_length, frame_step)
    windowed_signal = framed_signal * _hann_window()
    real_stft, imag_stft = _rdft(windowed_signal, fft_length)
    stft_magnitude = _complex_abs(real_stft, imag_stft)
    return stft_magnitude


# --- Custom Keras Layer for Feature Extraction with proper serialization ---
@tf.keras.utils.register_keras_serializable(package="yamnet", name="YamnetFeaturesLayer")
class YamnetFeaturesLayer(layers.Layer):
    def __init__(self, params=None, name="yamnet_features", **kwargs):
        super(YamnetFeaturesLayer, self).__init__(name=name, **kwargs)
        
        # If no params provided, create default
        if params is None:
            import params as yamnet_params_module
            params = yamnet_params_module.Params()
        
        self.params = params
        print(f"Initialized YamnetFeaturesLayer with sample_rate: {params.sample_rate}")

        # Store params as individual attributes for serialization
        self.mel_bands = params.mel_bands
        self.sample_rate = params.sample_rate
        self.mel_min_hz = params.mel_min_hz
        self.mel_max_hz = params.mel_max_hz
        self.stft_window_seconds = params.stft_window_seconds
        self.stft_hop_seconds = params.stft_hop_seconds
        self.patch_window_seconds = params.patch_window_seconds
        self.patch_hop_seconds = params.patch_hop_seconds
        self.log_offset = params.log_offset
        self.tflite_compatible = params.tflite_compatible

        # Pre-calculate Mel matrix - this is static and can be done once
        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.mel_bands,
            num_spectrogram_bins=self._calculate_num_spectrogram_bins(),
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.mel_min_hz,
            upper_edge_hertz=self.mel_max_hz)

    def _calculate_num_spectrogram_bins(self):
        """Helper to calculate FFT size based on params."""
        window_length_samples = int(round(self.sample_rate * self.stft_window_seconds))
        fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
        return fft_length // 2 + 1

    def _pad_waveform(self, waveform):
        """Internal padding logic, now part of the layer's call method."""
        input_shape = tf.shape(waveform)
        num_samples = input_shape[-1]

        min_waveform_seconds = (
            self.patch_window_seconds +
            self.stft_window_seconds - self.stft_hop_seconds)
        min_num_samples = tf.cast(min_waveform_seconds * self.sample_rate, tf.int32)
        num_padding_samples_min = tf.maximum(0, min_num_samples - num_samples)

        num_samples_eff = tf.maximum(num_samples, min_num_samples)
        num_samples_after_first_patch = num_samples_eff - min_num_samples
        hop_samples = tf.cast(self.patch_hop_seconds * self.sample_rate, tf.int32)
        hop_samples = tf.maximum(hop_samples, 1)
        num_hops_after_first_patch = tf.cast(tf.math.ceil(
                tf.cast(num_samples_after_first_patch, tf.float32) /
                tf.cast(hop_samples, tf.float32)), tf.int32)
        num_padding_samples_after_first = (
            hop_samples * num_hops_after_first_patch - num_samples_after_first_patch)
        total_padding_samples = num_padding_samples_min + num_padding_samples_after_first

        # Paddings assuming rank 2 input (Batch, Samples)
        rank = tf.rank(waveform)
        sample_padding = tf.stack([0, total_padding_samples])
        other_dims_padding = tf.zeros(shape=tf.stack([rank - 1, 2]), dtype=tf.int32)
        paddings = tf.concat([other_dims_padding, tf.expand_dims(sample_padding, axis=0)], axis=0)

        padded_waveform = tf.pad(waveform, paddings, mode='CONSTANT', constant_values=0.0)
        return padded_waveform

    def call(self, waveform):
        """Applies padding, STFT, Mel projection, log, and framing."""
        # 1. Pad waveform
        padded_waveform = self._pad_waveform(waveform)

        # 2. Compute STFT Magnitude Spectrogram
        window_length_samples = int(round(self.sample_rate * self.stft_window_seconds))
        hop_length_samples = int(round(self.sample_rate * self.stft_hop_seconds))
        fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))

        stft_tensor = tf.signal.stft(
            signals=padded_waveform,
            frame_length=window_length_samples,
            frame_step=hop_length_samples,
            fft_length=fft_length)
        magnitude_spectrogram = tf.abs(stft_tensor)

        # 3. Compute Mel Spectrogram
        mel_spectrogram = tf.matmul(magnitude_spectrogram, self.linear_to_mel_weight_matrix)

        # 4. Compute Log Mel Spectrogram
        log_mel_spectrogram = tf.math.log(mel_spectrogram + self.log_offset)

        # 5. Frame into Patches
        spectrogram_hop_length_samples = int(round(self.sample_rate * self.stft_hop_seconds))
        spectrogram_sample_rate = self.sample_rate / spectrogram_hop_length_samples
        patch_window_length_samples = int(round(spectrogram_sample_rate * self.patch_window_seconds))
        patch_hop_length_samples = int(round(spectrogram_sample_rate * self.patch_hop_seconds))

        # Apply framing along the time axis (axis=1 for batched input)
        features = tf.signal.frame(
            signal=log_mel_spectrogram,
            frame_length=patch_window_length_samples,
            frame_step=patch_hop_length_samples,
            axis=1)

        # Return both the full log-mel spectrogram and the framed features (patches)
        return log_mel_spectrogram, features

    def get_config(self):
        config = super(YamnetFeaturesLayer, self).get_config()
        config.update({
            "mel_bands": self.mel_bands,
            "sample_rate": self.sample_rate,
            "mel_min_hz": self.mel_min_hz,
            "mel_max_hz": self.mel_max_hz,
            "stft_window_seconds": self.stft_window_seconds,
            "stft_hop_seconds": self.stft_hop_seconds,
            "patch_window_seconds": self.patch_window_seconds,
            "patch_hop_seconds": self.patch_hop_seconds,
            "log_offset": self.log_offset,
            "tflite_compatible": self.tflite_compatible
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Recreate params object from config
        class ParamsFromConfig:
            def __init__(self, config):
                self.mel_bands = config.pop("mel_bands", 64)
                self.sample_rate = config.pop("sample_rate", 16000.0)
                self.mel_min_hz = config.pop("mel_min_hz", 125.0)
                self.mel_max_hz = config.pop("mel_max_hz", 7500.0)
                self.stft_window_seconds = config.pop("stft_window_seconds", 0.025)
                self.stft_hop_seconds = config.pop("stft_hop_seconds", 0.010)
                self.patch_window_seconds = config.pop("patch_window_seconds", 0.96)
                self.patch_hop_seconds = config.pop("patch_hop_seconds", 0.48)
                self.log_offset = config.pop("log_offset", 0.01)
                self.tflite_compatible = config.pop("tflite_compatible", False)
        
        params = ParamsFromConfig(config)
        return cls(params=params, **config)
