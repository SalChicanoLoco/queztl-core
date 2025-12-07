# Gen3D - AI 3D Model Generation App

AI-powered 3D model generation from text prompts and images. Completely standalone application.

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Access the app:**
- Frontend: http://localhost:3001
- Backend API: http://localhost:8001
- API Docs: http://localhost:8001/docs

### Manual Setup

#### Backend

```bash
cd backend
pip install -r requirements.txt
python main.py
```

Backend runs on http://localhost:8001

#### Frontend

```bash
cd frontend
# Serve with any web server, e.g.:
python -m http.server 3001
```

Frontend runs on http://localhost:3001

## 📋 Features

- **Text-to-3D Generation**: Create 3D models from text descriptions
- **Multiple Styles**: Realistic, Stylized, Low-Poly, Voxel
- **Detail Levels**: Low (500v), Medium (2K), High (8K), Ultra (32K)
- **Real-time 3D Viewer**: Interactive WebGL viewer with Three.js
- **Export Options**: Download as OBJ or JSON format
- **Fast Generation**: Models generated in 1-3 seconds

## 🎨 Usage Examples

### Text Prompts:
- "Futuristic spacecraft with sleek design"
- "Medieval castle tower with battlements"
- "Cyberpunk character with armor"
- "Ancient tree with twisted branches"
- "Modern skyscraper"
- "Crystal formation"

### API Usage:

```bash
# Generate model (GET)
curl "http://localhost:8001/api/text-to-3d?prompt=spacecraft&style=realistic&detail_level=medium&format=json"

# Get capabilities
curl "http://localhost:8001/api/capabilities"

# Health check
curl "http://localhost:8001/health"
```

## 🏗️ Architecture

```
gen3d-app/
├── backend/
│   ├── gen3d_engine.py    # 3D generation engine
│   ├── main.py            # FastAPI application
│   ├── requirements.txt   # Python dependencies
│   └── Dockerfile         # Backend container
├── frontend/
│   └── index.html         # Single-page application
├── docker-compose.yml     # Container orchestration
└── README.md             # This file
```

## 🔧 Configuration

### Ports
- Backend: 8001 (configurable in docker-compose.yml)
- Frontend: 3001 (configurable in docker-compose.yml)

### Detail Levels
- **Low**: 500 vertices (fast preview)
- **Medium**: 2,000 vertices (balanced)
- **High**: 8,000 vertices (detailed)
- **Ultra**: 32,000 vertices (maximum quality)

### Supported Styles
- Realistic: Natural proportions and details
- Stylized: Artistic interpretation
- Low-Poly: Faceted geometric style
- Voxel: Blocky Minecraft-like style

## 📦 Export Formats

### OBJ Format
- Standard 3D model format
- Compatible with Blender, Maya, 3ds Max
- Includes vertices, faces, normals

### JSON Format
- Three.js compatible
- Direct integration with web applications
- Includes vertices, faces, normals, UVs

## 🔌 API Endpoints

### GET/POST `/api/text-to-3d`
Generate 3D model from text prompt

**Parameters:**
- `prompt` (string): Description of model
- `style` (string): realistic|stylized|low-poly|voxel
- `detail_level` (string): low|medium|high|ultra
- `format` (string): json|obj

**Response:**
```json
{
  "model": {
    "vertices": [...],
    "faces": [...],
    "normals": [...]
  },
  "stats": {
    "vertices": 2000,
    "faces": 4000
  },
  "generation_time": 1.23
}
```

### POST `/api/image-to-3d`
Generate 3D model from image (coming soon)

### GET `/api/capabilities`
Get API capabilities and supported features

### GET `/health`
Health check endpoint

## 🛠️ Development

### Backend Development

```bash
cd backend
pip install -r requirements.txt

# Run with auto-reload
uvicorn main:app --reload --port 8001
```

### Frontend Development

Just edit `frontend/index.html` and refresh browser.

## 📊 Performance

- **Generation Speed**: 1-3 seconds per model
- **Memory Usage**: ~200MB per container
- **Concurrent Requests**: Handles 10+ simultaneous generations
- **Model Complexity**: Up to 32,000 vertices

## 🎯 Use Cases

- **Game Development**: Generate placeholder assets
- **3D Printing**: Create custom models
- **Education**: Learn 3D modeling concepts
- **Rapid Prototyping**: Quick design iterations
- **Art & Visualization**: Creative 3D art generation

## 🔄 Roadmap

- [ ] Image-to-3D conversion
- [ ] AI texture generation
- [ ] Multi-object scenes
- [ ] Animation support
- [ ] Model refinement tools
- [ ] Batch processing
- [ ] Cloud deployment templates

## 📝 License

This is part of the Queztl-Core project.

## 🆘 Support

For issues or questions:
1. Check API docs at http://localhost:8001/docs
2. Review logs: `docker-compose logs`
3. Check health: `curl http://localhost:8001/health`

---

**Built with ❤️ using FastAPI, Three.js, and NumPy**
