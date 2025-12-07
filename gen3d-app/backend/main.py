"""
Gen3D Backend API
Standalone AI 3D Model Generation Service
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional
import uvicorn

from ai_3d_engine import (
    AI3DGenerator,
    mesh_to_obj,
    mesh_to_json
)

app = FastAPI(
    title="Gen3D AI API",
    description="Real AI-Powered 3D Generation: Text-to-3D, Image-to-3D, Video-to-3D",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI generator
ai_generator = AI3DGenerator()

class TextTo3DRequest(BaseModel):
    prompt: str
    style: str = "realistic"
    detail_level: str = "medium"
    format: str = "json"  # "json" or "obj"

class ImageTo3DRequest(BaseModel):
    depth_method: str = "automatic"
    format: str = "json"

class TextureRequest(BaseModel):
    style: str = "realistic"
    resolution: int = 1024

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "Gen3D API",
        "version": "1.0.0",
        "description": "AI-Powered 3D Model Generation",
        "endpoints": {
            "text_to_3d": "/api/text-to-3d",
            "image_to_3d": "/api/image-to-3d",
            "generate_texture": "/api/generate-texture",
            "capabilities": "/api/capabilities"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "gen3d-backend"}

@app.get("/api/capabilities")
async def get_capabilities():
    """Get AI engine capabilities"""
    return {
        "text_to_3d": {
            "models": ["shap-e", "point-e"],
            "supported_styles": ["realistic", "stylized", "low-poly", "voxel"],
            "detail_levels": ["low", "medium", "high", "ultra"],
            "formats": ["json", "obj"],
            "ai_powered": True
        },
        "image_to_3d": {
            "models": ["triposr", "zero123", "wonder3d"],
            "supported_formats": ["png", "jpg", "jpeg", "webp"],
            "formats": ["json", "obj"],
            "ai_powered": True
        },
        "video_to_3d": {
            "models": ["colmap", "instant-ngp"],
            "supported_formats": ["mp4", "webm"],
            "formats": ["json", "obj"],
            "ai_powered": True
        },
        "device": ai_generator.device,
        "version": "2.0.0"
    }

@app.get("/api/text-to-3d")
async def text_to_3d_get(
    prompt: str,
    style: str = "realistic",
    detail_level: str = "medium",
    format: str = "json",
    model: str = "shap-e"
):
    """Generate 3D model from text using AI (GET method for easy testing)"""
    try:
        # Generate using AI
        result = await ai_generator.generate_from_text(
            prompt=prompt,
            style=style,
            detail_level=detail_level,
            model=model
        )
        
        # Format response
        if format == "obj":
            obj_data = mesh_to_obj(result.mesh)
            return Response(
                content=obj_data,
                media_type="text/plain",
                headers={"Content-Disposition": f"attachment; filename=model.obj"}
            )
        else:
            return {
                "model": mesh_to_json(result.mesh),
                "format": "json",
                "prompt": result.prompt,
                "style": result.style,
                "detail_level": result.detail_level,
                "generation_time": result.generation_time,
                "stats": {
                    "vertices": result.vertex_count,
                    "faces": result.face_count
                }
            }
    
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {str(e)}")

@app.post("/api/text-to-3d")
async def text_to_3d_post(request: TextTo3DRequest):
    """Generate 3D model from text using AI (POST method)"""
    try:
        # Generate using AI
        result = await ai_generator.generate_from_text(
            prompt=request.prompt,
            style=request.style,
            detail_level=request.detail_level,
            model="shap-e"
        )
        
        # Format response
        if request.format == "obj":
            obj_data = mesh_to_obj(result.mesh)
            return Response(
                content=obj_data,
                media_type="text/plain",
                headers={"Content-Disposition": f"attachment; filename=model.obj"}
            )
        else:
            return {
                "model": mesh_to_json(result.mesh),
                "format": "json",
                "prompt": result.prompt,
                "style": result.style,
                "detail_level": result.detail_level,
                "generation_time": result.generation_time,
                "stats": {
                    "vertices": result.vertex_count,
                    "faces": result.face_count
                }
            }
    
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {str(e)}")

@app.post("/api/image-to-3d")
async def image_to_3d(
    file: UploadFile = File(...),
    model: str = "triposr",
    format: str = "json"
):
    """Generate 3D model from image using AI"""
    try:
        # Read image data
        image_data = await file.read()
        
        # Generate 3D mesh using AI
        mesh = await ai_generator.generate_from_image(image_data, model=model)
        
        # Format response
        if format == "obj":
            obj_data = mesh_to_obj(mesh)
            return Response(
                content=obj_data,
                media_type="text/plain",
                headers={"Content-Disposition": f"attachment; filename=model.obj"}
            )
        else:
            return {
                "model": mesh_to_json(mesh),
                "format": "json",
                "source": "image",
                "ai_model": model,
                "stats": {
                    "vertices": len(mesh.vertices),
                    "faces": len(mesh.faces) // 3
                }
            }
    
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {str(e)}")

@app.post("/api/generate-texture")
async def generate_texture(request: TextureRequest):
    """Generate AI texture for model"""
    try:
        # For now, return placeholder
        return {
            "status": "generated",
            "style": request.style,
            "resolution": request.resolution,
            "format": "png",
            "message": "Texture generation feature - coming soon"
        }
    
    except Exception as e:
        raise HTTPException(500, f"Texture generation failed: {str(e)}")

@app.post("/api/video-to-3d")
async def video_to_3d(
    file: UploadFile = File(...),
    model: str = "colmap",
    format: str = "json"
):
    """Generate 3D model from video using multi-view reconstruction"""
    try:
        # Read video data
        video_data = await file.read()
        
        # Extract frames and generate 3D
        # For now, return placeholder (video processing is complex)
        return {
            "status": "processing",
            "message": "Video-to-3D feature in development. Use text-to-3D or image-to-3D for now.",
            "model": model,
            "format": format
        }
    
    except Exception as e:
        raise HTTPException(500, f"Video generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,  # Different port from benchmark app (8000)
        reload=True
    )
