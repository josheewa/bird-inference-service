---
title: Bird Sound Classification API
emoji: 🦅
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# 🦅 Bird Sound Classification API

A Flask-based REST API for bird sound classification using YAMNet + GRU ensemble model.

## Features

- **50 Bird Species Classification**: Trained on eBird audio data
- **YAMNet Backbone**: Google's audio classification model
- **REST API**: Easy integration with other applications
- **Audio Format Support**: WAV, MP3, FLAC, OGG, M4A, WebM

## API Endpoints

### GET `/health`
Check API health and model status

### GET `/species` 
Get list of all 50 supported bird species

### POST `/predict`
Upload audio file for bird species prediction
- **Body**: `multipart/form-data` with `audio` file
- **Response**: Top 5 predictions with confidence scores

### POST `/predict_url`
Predict from remote audio URL
- **Body**: `{"url": "https://example.com/audio.wav"}`
- **Response**: Top 5 predictions with confidence scores

## Usage

### Web Interface
Visit the deployed app URL to use the interactive web interface.

### API Usage
```bash
# Health check
curl https://your-space-url/health

# Get supported species
curl https://your-space-url/species

# Predict from file
curl -X POST -F "audio=@bird_sound.wav" https://your-space-url/predict

# Predict from URL
curl -X POST -H "Content-Type: application/json" \
     -d '{"url":"https://example.com/bird.wav"}' \
     https://your-space-url/predict_url
```

## Model Details

- **Architecture**: YAMNet + GRU (256 units) + Dense layers
- **Input**: 5-second audio segments at 16kHz
- **Output**: Softmax probabilities for 50 bird species
- **Model Size**: 56.1 MB

## Response Format

```json
{
  "success": true,
  "predictions": [
    {
      "rank": 1,
      "species": "Northern Cardinal",
      "scientific_name": "cardinalis cardinalis",
      "confidence": 0.92,
      "confidence_percent": 92.0
    }
  ],
  "processing_time_seconds": 0.156,
  "file_info": {
    "filename": "bird_sound.wav",
    "size_bytes": 1234567
  }
}
```

## Supported Species

The API can classify 50 common North American bird species:

- Northern Cardinal
- American Robin
- Blue Jay
- American Goldfinch
- House Finch
- Mourning Dove
- Carolina Chickadee
- Tufted Titmouse
- And 42 more...

## Audio Requirements

- **Duration:** Any length (automatically segmented to 5 seconds)
- **Sample Rate:** Any (automatically resampled to 16kHz)
- **Channels:** Mono or stereo (converted to mono)
- **Formats:** WAV, MP3, FLAC, OGG, M4A, WebM
- **Max Size:** 50MB

## Deployment

### Local Development
```bash
FLASK_ENV=development python run.py
```

### Production with Gunicorn
```bash
gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 app:app
```

### Environment Variables
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 5000)
- `FLASK_ENV`: Environment mode (development/production)

## Integration with Next.js

Here's how to integrate with your Next.js app:

```javascript
// utils/birdAPI.js
const API_BASE_URL = 'http://your-api-server:5000';

export async function classifyBirdSound(audioFile) {
  const formData = new FormData();
  formData.append('audio', audioFile);
  
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    body: formData,
  });
  
  return response.json();
}

export async function getSupportedSpecies() {
  const response = await fetch(`${API_BASE_URL}/species`);
  return response.json();
}
```

```jsx
// components/BirdClassifier.jsx
import { useState } from 'react';
import { classifyBirdSound } from '../utils/birdAPI';

export default function BirdClassifier() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    if (!file) return;
    
    setLoading(true);
    try {
      const result = await classifyBirdSound(file);
      setResults(result);
    } catch (error) {
      console.error('Prediction failed:', error);
    }
    setLoading(false);
  };

  return (
    <div>
      <input 
        type="file" 
        accept="audio/*"
        onChange={(e) => setFile(e.target.files[0])}
      />
      <button onClick={handlePredict} disabled={!file || loading}>
        {loading ? 'Processing...' : 'Classify Bird'}
      </button>
      
      {results && (
        <div>
          <h3>Results:</h3>
          {results.predictions.map((pred, i) => (
            <div key={i}>
              {pred.species} - {pred.confidence_percent}%
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid file, missing parameters)
- `413`: File too large
- `500`: Internal server error

Error responses include a descriptive message:
```json
{
  "error": "File type not supported. Allowed: ['wav', 'mp3', 'flac', 'ogg', 'm4a', 'webm']"
}
```

## Performance Tips

1. **File Size:** Smaller files process faster
2. **Audio Quality:** 16kHz mono is optimal
3. **Caching:** Results are not cached, implement client-side caching if needed
4. **Concurrent Requests:** API supports multiple concurrent requests

## Troubleshooting

### Model Not Loading
- Ensure `best_model.keras` exists in `../checkpoints_94/`
- Check that YAMNet files are in `../../yamnet/`
- Verify species.json exists in `../../species.json`

### Memory Issues
- Reduce batch size in production
- Monitor memory usage with large files
- Consider implementing request queuing for high load

### Slow Inference
- Check TensorFlow is using GPU if available
- Monitor CPU usage during inference
- Consider model quantization for faster inference

## License

This API is part of the Bird Sound Classification project. 