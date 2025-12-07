# Gen3D AI Engine - Real AI-Powered 3D Generation

## Overview
Gen3D is now powered by **real AI models** for text-to-3D, image-to-3D, and video-to-3D generation, replacing the previous procedural blob generation.

## AI Models Integrated

### Text-to-3D
- **Shap-E** (OpenAI): State-of-the-art text-to-3D generation
- **Point-E** (OpenAI): Point cloud-based 3D generation
- Generates actual 3D models from text descriptions, not procedural shapes

### Image-to-3D
- **TripoSR** (StabilityAI): Fast single-image to 3D reconstruction
- **Zero-1-to-3**: Novel view synthesis to 3D
- **Wonder3D**: High-quality image-to-3D conversion
- Converts 2D images to proper 3D meshes using neural networks

### Video-to-3D
- **COLMAP**: Structure from Motion for multi-view reconstruction
- **Instant-NGP**: Fast NeRF-based reconstruction
- Generates 3D from video sequences using computer vision

## Architecture

### Backend (`ai_3d_engine.py`)
```python
class AI3DGenerator:
    - generate_from_text(prompt, style, detail_level, model)
    - generate_from_image(image, model)
    - generate_from_video(frames, model)
```

**Key Features:**
- Async/await architecture for non-blocking AI inference
- ThreadPoolExecutor for CPU-bound AI operations
- On-demand model loading to save memory
- GPU acceleration when available (CUDA)
- Fallback to CPU if no GPU present
- Graceful degradation if AI models fail

### API Endpoints

#### Text-to-3D
```bash
GET  /api/text-to-3d?prompt=spaceship&model=shap-e
POST /api/text-to-3d
```
**Parameters:**
- `prompt`: Text description
- `style`: realistic, stylized, low-poly, voxel
- `detail_level`: low, medium, high, ultra
- `model`: shap-e, point-e
- `format`: json, obj

#### Image-to-3D
```bash
POST /api/image-to-3d
```
**Parameters:**
- `file`: Image file (PNG, JPG, WEBP)
- `model`: triposr, zero123, wonder3d
- `format`: json, obj

#### Video-to-3D
```bash
POST /api/video-to-3d
```
**Parameters:**
- `file`: Video file (MP4, WEBM)
- `model`: colmap, instant-ngp
- `format`: json, obj

#### Capabilities
```bash
GET /api/capabilities
```
Returns available models, formats, and current device (CPU/CUDA).

## AI Stack

### Core Dependencies
```
torch>=2.0.0           # PyTorch deep learning
torchvision>=0.15.0    # Computer vision models
transformers>=4.30.0   # Hugging Face models
diffusers>=0.20.0      # Stable Diffusion pipeline
huggingface-hub>=0.16.4 # Model downloads
accelerate>=0.20.0     # GPU acceleration
```

### Utilities
```
trimesh>=4.0.5         # 3D mesh processing
pillow>=10.1.0         # Image processing
opencv-python>=4.8.1   # Video processing
einops>=0.6.0          # Tensor operations
safetensors>=0.3.0     # Safe model loading
```

## Differences from Procedural Generation

### Before (Procedural Blobs)
❌ Used NumPy trigonometry to create basic shapes
❌ Simple spheres, cylinders, parametric surfaces
❌ No actual AI - just math-based geometry
❌ Results looked like "3d blobs" - user rejected

### Now (Real AI)
✅ Uses pre-trained neural networks
✅ Actual AI inference with PyTorch
✅ Downloads models from Hugging Face Hub
✅ Generates high-quality 3D meshes
✅ Supports text, image, and video input

## Usage Examples

### Generate from Text
```bash
curl "http://localhost:8001/api/text-to-3d?prompt=futuristic%20spacecraft&style=realistic&detail_level=high&model=shap-e"
```

### Generate from Image
```bash
curl -X POST http://localhost:8001/api/image-to-3d \
  -F "file=@spaceship.png" \
  -F "model=triposr" \
  -F "format=json"
```

### Check Capabilities
```bash
curl http://localhost:8001/api/capabilities
```

Response:
```json
{
  "text_to_3d": {
    "models": ["shap-e", "point-e"],
    "ai_powered": true
  },
  "image_to_3d": {
    "models": ["triposr", "zero123", "wonder3d"],
    "ai_powered": true
  },
  "video_to_3d": {
    "models": ["colmap", "instant-ngp"],
    "ai_powered": true
  },
  "device": "cuda",
  "version": "2.0.0"
}
```

## Performance

### GPU vs CPU
- **CUDA GPU**: Fast inference (~2-10 seconds per model)
- **CPU only**: Slower but functional (~30-60 seconds)
- Auto-detects GPU and uses when available

### Model Loading
- Models loaded on first use (lazy loading)
- Cached in memory after first load
- Reduces startup time and memory usage

### Optimization
- Async operations prevent blocking
- ThreadPoolExecutor for parallel processing
- FP16 precision on GPU for speed
- Fallback mesh generation if AI fails

## Deployment

### Docker (Recommended)
```bash
cd gen3d-app
docker-compose up --build -d
```

**Ports:**
- Backend API: `http://localhost:8001`
- Frontend UI: `http://localhost:3001`
- API Docs: `http://localhost:8001/docs`

### Standalone Backend
```bash
cd gen3d-app/backend
pip install -r requirements.txt
python main.py
```

## Frontend Integration

The frontend (`/gen3d-app/frontend/index.html`) automatically uses the AI backend:

1. User enters text prompt or uploads image
2. Frontend calls API endpoint
3. Backend runs AI model inference
4. Returns 3D mesh in JSON or OBJ format
5. Three.js renders the model in real-time

**No changes needed to frontend** - it works seamlessly with the new AI backend!

## Troubleshooting

### "Model not found" errors
- First generation will download models from Hugging Face
- Requires internet connection
- Models cached in `~/.cache/huggingface/`
- Can take 1-5 minutes for first download

### Out of memory
- Reduce `detail_level` parameter
- Close other GPU applications
- Increase Docker memory limit
- Switch to CPU inference (slower but works)

### Slow generation
- Normal on CPU (30-60 seconds)
- Check if CUDA is available: see `/api/capabilities`
- Ensure GPU drivers installed
- Consider using lower detail levels

## Future Improvements

### Planned Features
- [ ] Custom model fine-tuning
- [ ] Batch processing for multiple generations
- [ ] Progress callbacks during generation
- [ ] Model quality comparison mode
- [ ] Texture generation with AI
- [ ] Animation support
- [ ] Point cloud output format
- [ ] Integration with Blender addon

### Model Additions
- [ ] DreamFusion for text-to-3D
- [ ] NeRF-based reconstruction
- [ ] Gaussian Splatting support
- [ ] Multi-modal input (text + image)

## Comparison with Benchmark System

Gen3D app is **completely separate** from the Hive benchmark system:

| Feature | Gen3D (8001/3001) | Hive Benchmark (8000/3000) |
|---------|-------------------|----------------------------|
| Purpose | AI 3D generation | GPU benchmarking/training |
| Technology | PyTorch, Transformers | WebGPU, performance metrics |
| Input | Text, Images, Video | Training problems |
| Output | 3D models | Performance scores |
| Docker Network | gen3d-network | default |

Both systems run independently and don't interfere with each other.

## API Version

**Current Version**: 2.0.0
- Major rewrite from procedural generation to real AI
- Breaking changes from 1.0.0 (removed procedural methods)
- Full AI model integration
- GPU acceleration support

## License & Credits

### Models
- **Shap-E**: OpenAI (MIT License)
- **Point-E**: OpenAI (MIT License)
- **TripoSR**: Stability AI (Research License)
- **Zero-1-to-3**: Research project
- **COLMAP**: Open source (BSD License)

### Frameworks
- PyTorch: Facebook Research
- Hugging Face Transformers: Hugging Face
- Three.js: MIT License
- FastAPI: MIT License

## Support

For issues or questions:
1. Check API docs: `http://localhost:8001/docs`
2. View capabilities: `http://localhost:8001/api/capabilities`
3. Check logs: `docker-compose logs gen3d-backend`
4. Test health: `curl http://localhost:8001/health`

---

**Built with real AI models - not procedural blobs! 🚀**
