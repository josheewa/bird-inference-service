#!/usr/bin/env python3
"""
Bird Sound Classification API Service

A Flask-based REST API for bird sound classification using the trained model.
Designed to be deployed and queried from external applications.

Version: 2.1.2 - Enhanced model weight tracking and robust loading fixes
"""

import os
import sys
import json
import time
import tempfile
import traceback
from io import BytesIO
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import warnings

# Print version info for deployment tracking
API_VERSION = "2.1.2"
print(f"🚀 Bird Sound Classification API v{API_VERSION}")
print(f"📅 Deployment timestamp: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}")

# Suppress warnings and TensorFlow logs
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Fix mixed precision policy issue for production deployment
print("🔧 Setting up TensorFlow mixed precision policy...")
try:
    # Clear any existing policy first
    tf.keras.backend.clear_session()
    
    # Explicitly set to float32 policy to avoid mixed precision issues
    policy = tf.keras.mixed_precision.Policy('float32')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("   ✅ Set global policy to float32")
    
    # Verify the policy was set correctly
    current_policy = tf.keras.mixed_precision.global_policy()
    print(f"   ✅ Current global policy: {current_policy}")
    
except Exception as e:
    print(f"   ⚠️ Could not set mixed precision policy: {e}")

# Configuration
class Config:
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.keras")
    SPECIES_PATH = os.path.join(os.path.dirname(__file__), "species.json")
    YAMNET_PATH = os.path.join(os.path.dirname(__file__), "yamnet")
    
    SAMPLE_RATE = 16000
    SEGMENT_DURATION = 5
    EXPECTED_SAMPLES = int(SAMPLE_RATE * SEGMENT_DURATION)
    
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'ogg', 'm4a', 'webm'}
    
    # API Configuration
    MAX_CONFIDENCE_RESULTS = 5  # Return top 5 predictions

# Add YAMNet to path
sys.path.insert(0, Config.YAMNET_PATH)

try:
    import params as yamnet_params_module
    params = yamnet_params_module.Params()
    import yamnet
    import features as features_lib
except ImportError as e:
    print(f"❌ Error importing YAMNet: {e}")
    print(f"Make sure yamnet directory exists at: {Config.YAMNET_PATH}")
    sys.exit(1)

# Model class (same as training)
@tf.keras.utils.register_keras_serializable()
class BirdClassifier(tf.keras.Model):
    def __init__(self, num_classes, yamnet_weights_path_arg=None, yamnet_trainable=True,
                 gru_units=256, dense_units=512, dropout_rate=0.5, l2_reg=0.005, **kwargs):
        name_kwarg = kwargs.pop('name', 'BirdClassifier')
        super(BirdClassifier, self).__init__(name=name_kwarg, **kwargs)
        self.num_classes = num_classes
        self.yamnet_trainable = yamnet_trainable
        self._gru_units = gru_units
        self._dense_units = dense_units
        self._dropout_rate = dropout_rate
        self._l2_reg = l2_reg
        self._yamnet_weights_path_ref = yamnet_weights_path_arg
        
        self.yamnet = yamnet.yamnet_frames_model(params=params)
        
        if self._yamnet_weights_path_ref and os.path.exists(self._yamnet_weights_path_ref):
            try:
                self.yamnet.load_weights(self._yamnet_weights_path_ref, by_name=True)
            except Exception as e:
                print(f"Warning: Could not load YAMNet weights: {e}")
        
        self.yamnet.trainable = self.yamnet_trainable
        
        self.gru = tf.keras.layers.GRU(gru_units, return_sequences=False, name="gru_layer")
        self.bn1 = tf.keras.layers.BatchNormalization(name="bn_1")
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate, name="dropout_1")
        self.dense1 = tf.keras.layers.Dense(dense_units, activation='relu',
                                          kernel_regularizer=tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None,
                                          name="dense_1")
        self.bn2 = tf.keras.layers.BatchNormalization(name="bn_2")
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate, name="dropout_2")
        self.output_dense = tf.keras.layers.Dense(num_classes, activation='softmax', name="output_layer")

    def call(self, inputs, training=None):
        yamnet_outputs = self.yamnet(inputs, training=(self.yamnet.trainable and training))
        _predictions, embeddings, _log_mel = yamnet_outputs
        x = self.gru(embeddings, training=training)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        x = self.dense1(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)
        return self.output_dense(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'yamnet_weights_path_arg': self._yamnet_weights_path_ref,
            'yamnet_trainable': self.yamnet_trainable,
            'gru_units': self._gru_units,
            'dense_units': self._dense_units,
            'dropout_rate': self._dropout_rate,
            'l2_reg': self._l2_reg
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Extract custom arguments
        custom_args_keys = ['num_classes', 'yamnet_weights_path_arg', 'yamnet_trainable', 
                           'gru_units', 'dense_units', 'dropout_rate', 'l2_reg']
        custom_args = {}
        remaining_config = {}
        
        for key, value in config.items():
            if key in custom_args_keys:
                custom_args[key] = value
            else:
                remaining_config[key] = value
        
        # Handle yamnet weights path
        saved_weights_basename = custom_args.pop('yamnet_weights_path_arg', None)
        if saved_weights_basename:
            custom_args['yamnet_weights_path_arg'] = os.path.join(Config.YAMNET_PATH, "yamnet.h5")
        else:
            custom_args['yamnet_weights_path_arg'] = os.path.join(Config.YAMNET_PATH, "yamnet.h5")
        
        return cls(**custom_args, **remaining_config)

# Global variables for model and species mapping
model = None
model_weights_ok = False  # Track if weights were loaded successfully
species_list = []
species_mapping = {}

def load_model():
    """Load the trained model with enhanced error handling for production deployment"""
    global model, model_weights_ok
    
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {Config.MODEL_PATH}")
    
    print(f"Loading model from: {Config.MODEL_PATH}")
    print(f"Model file size: {os.path.getsize(Config.MODEL_PATH) / (1024*1024):.1f} MB")
    
    # Initialize flags
    model = None
    model_weights_ok = False
    
    # Defensively get YamnetFeaturesLayer to avoid AttributeError
    yamnet_features_layer = getattr(features_lib, 'YamnetFeaturesLayer', None)
    custom_objects = {'BirdClassifier': BirdClassifier}
    if yamnet_features_layer is not None:
        custom_objects['YamnetFeaturesLayer'] = yamnet_features_layer
    
    # Strategy 1: Try loading with explicit policy management
    print("🔄 Trying load with explicit policy management...")
    try:
        # Clear session and set policy again
        tf.keras.backend.clear_session()
        
        # Force float32 policy context
        with tf.keras.mixed_precision.policy_scope('float32'):
            model = tf.keras.models.load_model(
                Config.MODEL_PATH,
                custom_objects=custom_objects,
                compile=False  # Skip compilation to avoid policy issues
            )
            
            print("✅ Model loaded successfully with explicit policy management!")
            model_weights_ok = True  # Weights loaded with the model
            
            # Test the model
            print("🧪 Testing model with dummy input...")
            dummy_input = tf.zeros((1, Config.EXPECTED_SAMPLES))
            dummy_output = model(dummy_input, training=False)
            print(f"   Output shape: {dummy_output.shape}")
            print(f"   Output sum: {tf.reduce_sum(dummy_output).numpy():.6f}")
            
            return True
            
    except Exception as e:
        print(f"❌ Policy management approach failed: {e}")
        if "str" in str(e) and "name" in str(e):
            print("   This is the mixed precision 'str' object error")
        model = None
        model_weights_ok = False
    
    # Strategy 2: Try simple load without custom objects or compilation
    print("🔄 Trying simple load (no custom objects, no compilation)...")
    try:
        tf.keras.backend.clear_session()
        
        # Try to load as a basic Keras model
        with tf.keras.mixed_precision.policy_scope('float32'):
            model = tf.keras.models.load_model(Config.MODEL_PATH, compile=False)
            
            print("✅ Simple load successful!")
            model_weights_ok = True  # Weights loaded with the model
            
            # Test the model
            print("🧪 Testing simple loaded model...")
            dummy_input = tf.zeros((1, Config.EXPECTED_SAMPLES))
            dummy_output = model(dummy_input, training=False)
            print(f"   Output shape: {dummy_output.shape}")
            print(f"   Output sum: {tf.reduce_sum(dummy_output).numpy():.6f}")
            
            return True
            
    except Exception as e:
        print(f"❌ Simple load failed: {e}")
        model = None
        model_weights_ok = False
    
    # Strategy 3: Manual model recreation with weights loading
    print("🔄 Trying manual model recreation...")
    try:
        # Clear session completely
        tf.keras.backend.clear_session()
        
        # Force explicit float32 context
        with tf.keras.mixed_precision.policy_scope('float32'):
            # Create new model instance
            yamnet_weights_path = os.path.join(Config.YAMNET_PATH, "yamnet.h5")
            model = BirdClassifier(
                num_classes=50,  # Known from our dataset
                yamnet_weights_path_arg=yamnet_weights_path,
                yamnet_trainable=False  # Set to False for inference
            )
            
            # Build the model
            dummy_input = tf.zeros((1, Config.EXPECTED_SAMPLES))
            _ = model(dummy_input, training=False)
            
            print("   ✅ Created new model instance")
            print("   🔄 Loading weights from saved model...")
            
            # Load only the weights from the saved model
            try:
                # Load model architecture and get weights WITH custom objects
                temp_model = tf.keras.models.load_model(
                    Config.MODEL_PATH, 
                    custom_objects=custom_objects,
                    compile=False
                )
                
                # Get the weights from the BirdClassifier layer if it exists
                weights_loaded = False
                for layer in temp_model.layers:
                    if hasattr(layer, 'name') and 'bird_classifier' in layer.name.lower():
                        # This is likely our custom model layer
                        print(f"   Found BirdClassifier layer: {layer.name}")
                        if hasattr(layer, 'get_weights') and layer.get_weights():
                            try:
                                model.set_weights(layer.get_weights())
                                weights_loaded = True
                                print("   ✅ Loaded weights from BirdClassifier layer")
                                break
                            except Exception as weight_error:
                                print(f"   ⚠️ Could not load weights: {weight_error}")
                
                # Fallback: try to load weights by matching layer names
                if not weights_loaded:
                    print("   🔄 Trying layer-by-layer weight transfer...")
                    weights_transferred = 0
                    for our_layer in model.layers:
                        try:
                            if hasattr(our_layer, 'name'):
                                matching_layer = temp_model.get_layer(our_layer.name)
                                if matching_layer.weights and our_layer.weights:
                                    our_layer.set_weights(matching_layer.get_weights())
                                    weights_transferred += 1
                        except:
                            continue
                    
                    if weights_transferred > 0:
                        print(f"   ✅ Transferred weights for {weights_transferred} layers")
                        weights_loaded = True
                
                del temp_model  # Clean up
                
                if weights_loaded:
                    print("✅ Model recreation with weights successful!")
                    model_weights_ok = True  # Weights successfully loaded
                    
                    # Test the model
                    print("🧪 Testing recreated model...")
                    test_output = model(dummy_input, training=False)
                    print(f"   Output shape: {test_output.shape}")
                    print(f"   Output sum: {tf.reduce_sum(test_output).numpy():.6f}")
                    
                    return True
                else:
                    print("⚠️ Model created but no weights loaded - using random weights")
                    model_weights_ok = False  # No weights loaded
                    return True  # Model structure exists but no proper weights
                    
            except Exception as weight_error:
                print(f"   ❌ Weight loading failed: {weight_error}")
                print("   ⚠️ Using model with initialized weights (emergency fallback)")
                model_weights_ok = False  # No weights loaded
                return True  # Model structure exists but no proper weights
                
    except Exception as e:
        print(f"❌ Manual model recreation failed: {e}")
        model = None
        model_weights_ok = False
    
    # Strategy 4: Emergency fallback - create a functioning model with random weights
    print("🔄 Emergency fallback: Creating model with random weights...")
    try:
        tf.keras.backend.clear_session()
        
        with tf.keras.mixed_precision.policy_scope('float32'):
            yamnet_weights_path = os.path.join(Config.YAMNET_PATH, "yamnet.h5")
            model = BirdClassifier(
                num_classes=50,
                yamnet_weights_path_arg=yamnet_weights_path,
                yamnet_trainable=False
            )
            
            # Build the model
            dummy_input = tf.zeros((1, Config.EXPECTED_SAMPLES))
            _ = model(dummy_input, training=False)
            
            print("⚠️ Emergency fallback model created (random weights)")
            print("⚠️ This model will give random predictions but the API will function")
            model_weights_ok = False  # Explicitly mark weights as not loaded
            
            return True
            
    except Exception as e:
        print(f"❌ Emergency fallback failed: {e}")
        model = None
        model_weights_ok = False
    
    print("❌ All model loading strategies failed")
    return False

def load_species_data():
    """Load species mapping"""
    global species_list, species_mapping
    
    if not os.path.exists(Config.SPECIES_PATH):
        print(f"Warning: Species file not found at: {Config.SPECIES_PATH}")
        return False
    
    try:
        with open(Config.SPECIES_PATH, 'r') as f:
            species_data = json.load(f)
        
        species_list = [item['name'] for item in species_data]
        
        # Create mapping from index to species info
        species_mapping = {}
        for i, item in enumerate(species_data):
            species_mapping[i] = {
                'name': item['name'],
                'scientific_name': item['query']
            }
        
        print(f"✅ Loaded {len(species_list)} species")
        return True
    except Exception as e:
        print(f"❌ Error loading species data: {e}")
        return False

def preprocess_audio(audio_data, sample_rate=None):
    """Preprocess audio for model inference"""
    try:
        # If audio_data is bytes, convert to numpy array
        if isinstance(audio_data, bytes):
            # Save to temporary file and load with librosa
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_file.flush()
                audio, sr = librosa.load(tmp_file.name, sr=Config.SAMPLE_RATE, mono=True)
                os.unlink(tmp_file.name)
        else:
            # Assume it's already a numpy array
            audio = audio_data
            if sample_rate and sample_rate != Config.SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=Config.SAMPLE_RATE)
        
        # Handle empty audio
        if len(audio) == 0:
            audio = np.zeros(Config.EXPECTED_SAMPLES, dtype=np.float32)
        
        # Pad or trim to expected length
        if len(audio) > Config.EXPECTED_SAMPLES:
            audio = audio[:Config.EXPECTED_SAMPLES]
        elif len(audio) < Config.EXPECTED_SAMPLES:
            audio = np.pad(audio, (0, Config.EXPECTED_SAMPLES - len(audio)), mode='constant')
        
        return audio.astype(np.float32)
    
    except Exception as e:
        print(f"Error preprocessing audio: {e}")
        return None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

@app.route('/')
def index():
    """Landing page with API documentation"""
    return render_template('index.html', species_count=len(species_list))

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_weights_ok': model_weights_ok,
        'species_count': len(species_list),
        'timestamp': time.time()
    })

@app.route('/species')
def get_species():
    """Get list of all supported species"""
    return jsonify({
        'species': species_list,
        'count': len(species_list)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    start_time = time.time()
    
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Check if model weights are properly loaded
        if not model_weights_ok:
            return jsonify({
                'error': 'Model weights not loaded properly - predictions would be random',
                'details': 'The model structure exists but weights were not loaded successfully'
            }), 500
        
        # Check if file was uploaded
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'File type not supported. Allowed: {list(Config.ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Check file size
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > Config.MAX_FILE_SIZE:
            return jsonify({
                'error': f'File too large. Max size: {Config.MAX_FILE_SIZE / (1024*1024):.1f}MB'
            }), 400
        
        # Read and preprocess audio
        audio_data = file.read()
        audio = preprocess_audio(audio_data)
        
        if audio is None:
            return jsonify({'error': 'Failed to process audio file'}), 400
        
        # Make prediction
        audio_batch = np.expand_dims(audio, axis=0)
        prediction = model.predict(audio_batch, verbose=0)
        
        # Get top predictions
        top_indices = np.argsort(prediction[0])[::-1][:Config.MAX_CONFIDENCE_RESULTS]
        
        predictions = []
        for i, idx in enumerate(top_indices):
            confidence = float(prediction[0][idx])
            species_info = species_mapping.get(idx, {'name': f'Unknown_{idx}', 'scientific_name': 'Unknown'})
            
            predictions.append({
                'rank': i + 1,
                'species': species_info['name'],
                'scientific_name': species_info['scientific_name'],
                'confidence': confidence,
                'confidence_percent': round(confidence * 100, 2)
            })
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'processing_time_seconds': round(processing_time, 3),
            'file_info': {
                'filename': secure_filename(file.filename),
                'size_bytes': file_size
            }
        })
    
    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return jsonify({
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/predict_url', methods=['POST'])
def predict_url():
    """Predict from audio URL (for remote files)"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Check if model weights are properly loaded
        if not model_weights_ok:
            return jsonify({
                'error': 'Model weights not loaded properly - predictions would be random',
                'details': 'The model structure exists but weights were not loaded successfully'
            }), 500
        
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'error': 'No URL provided'}), 400
        
        url = data['url']
        
        # Download and process audio from URL
        import requests
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Check content length
        content_length = int(response.headers.get('content-length', 0))
        if content_length > Config.MAX_FILE_SIZE:
            return jsonify({
                'error': f'File too large. Max size: {Config.MAX_FILE_SIZE / (1024*1024):.1f}MB'
            }), 400
        
        # Read audio data
        audio_data = response.content
        
        # Process same as file upload
        audio = preprocess_audio(audio_data)
        if audio is None:
            return jsonify({'error': 'Failed to process audio from URL'}), 400
        
        # Make prediction (same logic as above)
        audio_batch = np.expand_dims(audio, axis=0)
        prediction = model.predict(audio_batch, verbose=0)
        
        top_indices = np.argsort(prediction[0])[::-1][:Config.MAX_CONFIDENCE_RESULTS]
        
        predictions = []
        for i, idx in enumerate(top_indices):
            confidence = float(prediction[0][idx])
            species_info = species_mapping.get(idx, {'name': f'Unknown_{idx}', 'scientific_name': 'Unknown'})
            
            predictions.append({
                'rank': i + 1,
                'species': species_info['name'],
                'scientific_name': species_info['scientific_name'],
                'confidence': confidence,
                'confidence_percent': round(confidence * 100, 2)
            })
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'url': url
        })
    
    except Exception as e:
        print(f"URL prediction error: {e}")
        return jsonify({
            'error': f'Failed to process URL: {str(e)}'
        }), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

def initialize_app():
    """Initialize the application"""
    print(f"🦅 Bird Sound Classification API v{API_VERSION}")
    print("=" * 50)
    
    # Print environment info for debugging
    print(f"🔧 Environment info:")
    print(f"   Python: {sys.version}")
    print(f"   TensorFlow: {tf.__version__}")
    print(f"   Working directory: {os.getcwd()}")
    print(f"   PYTHONPATH: {':'.join(sys.path[:3])}...")
    
    # Check if files exist
    print(f"📁 File checks:")
    print(f"   Model file: {'✅' if os.path.exists(Config.MODEL_PATH) else '❌'} {Config.MODEL_PATH}")
    print(f"   Species file: {'✅' if os.path.exists(Config.SPECIES_PATH) else '❌'} {Config.SPECIES_PATH}")
    print(f"   YAMNet dir: {'✅' if os.path.exists(Config.YAMNET_PATH) else '❌'} {Config.YAMNET_PATH}")
    
    # Load model
    if not load_model():
        print("❌ Failed to load model")
        return False
    
    # Load species data
    if not load_species_data():
        print("❌ Failed to load species data")
        return False
    
    print(f"✅ API v{API_VERSION} initialized successfully!")
    print(f"   Model: {Config.MODEL_PATH}")
    print(f"   Species: {len(species_list)} supported")
    return True

if __name__ == '__main__':
    if initialize_app():
        print("\n🚀 Starting Flask server...")
        print("📡 API endpoints:")
        print("   GET  /health - Health check")
        print("   GET  /species - List supported species")
        print("   POST /predict - Upload audio file for prediction")
        print("   POST /predict_url - Predict from audio URL")
        print("")
        
        # For Render deployment, use the PORT environment variable
        port = int(os.environ.get('PORT', 5000))
        
        app.run(
            host='0.0.0.0',
            port=port,
            debug=os.environ.get('FLASK_ENV') == 'development'
        )
    else:
        print("❌ Failed to initialize app")
        sys.exit(1)

# Auto-initialize for production deployment (Gunicorn)
else:
    print("🔄 Initializing app for production...")
    try:
        if not initialize_app():
            print("❌ Failed to initialize app for production")
            raise RuntimeError("Model initialization failed")
        else:
            print("✅ Production initialization complete")
    except Exception as e:
        print(f"❌ Production initialization error: {e}")
        # Don't raise the error immediately - let the app start and show error in health endpoint
        print("⚠️ Continuing with partial initialization...")
        pass 