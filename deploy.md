# Deployment Instructions for Render

## Files Structure for Deployment

Your `api_service` directory now contains all necessary files for deployment:

```
api_service/
├── app.py                 # Main Flask application
├── run.py                 # Local development runner
├── requirements.txt       # Python dependencies
├── Procfile              # Render deployment config
├── species.json          # Bird species data
├── best_model.keras      # Trained model (56MB)
├── yamnet/               # YAMNet model files
│   ├── __init__.py
│   ├── features.py
│   ├── params.py
│   ├── yamnet.py
│   └── yamnet.h5         # YAMNet weights (15MB)
└── templates/
    └── index.html        # Web interface
```

## Render Deployment Steps

1. **Push to GitHub**
   - Copy the entire `api_service` directory to a new GitHub repository
   - Ensure all files including `best_model.keras` and `yamnet.h5` are included

2. **Create Render Service**
   - Go to [render.com](https://render.com)
   - Connect your GitHub repository
   - Choose "Web Service"

3. **Configure Render Settings**
   ```
   Name: bird-sound-api
   Environment: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
   ```

4. **Environment Variables** (optional)
   ```
   FLASK_ENV=production
   ```

## Important Notes

- **File Size**: The repository will be ~71MB due to model files
- **Memory**: Ensure you have at least 2GB RAM allocated
- **Timeout**: Set request timeout to at least 120 seconds for model loading
- **Cold Starts**: First request may take 10-30 seconds as model loads

## Testing After Deployment

Once deployed, test your API:

```bash
# Replace YOUR_RENDER_URL with your actual Render URL
curl https://YOUR_RENDER_URL.onrender.com/health
curl https://YOUR_RENDER_URL.onrender.com/species
```

## Troubleshooting

### Model Loading Issues
- Ensure `best_model.keras` is in the root directory
- Check that all yamnet files are present in `yamnet/` folder

### Memory Issues
- Upgrade to a plan with more RAM if needed
- Consider model quantization for smaller memory footprint

### Timeout Issues
- Increase worker timeout in Procfile
- Optimize model loading for faster cold starts

## Git LFS for Large Files (Optional)

If you have issues with large files, use Git LFS:

```bash
git lfs install
git lfs track "*.keras"
git lfs track "*.h5"
git add .gitattributes
git add .
git commit -m "Add model files with LFS"
git push
``` 