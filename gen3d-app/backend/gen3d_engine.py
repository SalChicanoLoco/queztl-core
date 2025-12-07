"""
Gen3D Engine - AI-Powered 3D Model Generation
Procedural geometry generation from text prompts and images
"""
import numpy as np
import asyncio
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time
import hashlib
import json
import base64

@dataclass
class Mesh3D:
    """3D mesh representation"""
    vertices: np.ndarray  # Nx3 array of vertex positions
    faces: np.ndarray     # Mx3 array of triangle indices
    normals: np.ndarray   # Nx3 array of vertex normals
    uvs: Optional[np.ndarray] = None  # Nx2 array of texture coordinates
    colors: Optional[np.ndarray] = None  # Nx3 array of vertex colors

@dataclass
class Generation3DResult:
    """Result of 3D generation"""
    mesh: Mesh3D
    prompt: str
    style: str
    detail_level: str
    generation_time: float
    vertex_count: int
    face_count: int

class TextTo3DGenerator:
    """Generate 3D models from text prompts"""
    
    def __init__(self):
        self.supported_styles = ["realistic", "stylized", "low-poly", "voxel"]
        self.detail_levels = {
            "low": 1500,      # Increased from 500
            "medium": 5000,   # Increased from 2000
            "high": 15000,    # Increased from 8000
            "ultra": 50000    # Increased from 32000
        }
    
    async def generate(self, prompt: str, style: str = "realistic", 
                      detail_level: str = "medium") -> Generation3DResult:
        """Generate 3D model from text prompt"""
        start_time = time.time()
        
        # Classify prompt to determine object type
        object_type = self._classify_prompt(prompt)
        
        # Get target vertex count
        target_vertices = self.detail_levels.get(detail_level, 2000)
        
        # Generate geometry based on type
        if object_type == "vehicle":
            mesh = self._generate_vehicle(prompt, target_vertices, style)
        elif object_type == "character":
            mesh = self._generate_character(prompt, target_vertices, style)
        elif object_type == "building":
            mesh = self._generate_building(prompt, target_vertices, style)
        elif object_type == "organic":
            mesh = self._generate_organic(prompt, target_vertices, style)
        else:
            mesh = self._generate_generic(prompt, target_vertices, style)
        
        generation_time = time.time() - start_time
        
        return Generation3DResult(
            mesh=mesh,
            prompt=prompt,
            style=style,
            detail_level=detail_level,
            generation_time=generation_time,
            vertex_count=len(mesh.vertices),
            face_count=len(mesh.faces)
        )
    
    def _classify_prompt(self, prompt: str) -> str:
        """Classify prompt to determine what type of object to generate"""
        prompt_lower = prompt.lower()
        
        # Vehicle keywords
        if any(word in prompt_lower for word in ['car', 'truck', 'plane', 'spacecraft', 
                                                   'ship', 'vehicle', 'boat', 'aircraft']):
            return "vehicle"
        
        # Character keywords
        if any(word in prompt_lower for word in ['person', 'character', 'human', 'robot',
                                                   'creature', 'monster', 'hero', 'warrior']):
            return "character"
        
        # Building keywords
        if any(word in prompt_lower for word in ['building', 'house', 'tower', 'castle',
                                                   'structure', 'architecture', 'skyscraper']):
            return "building"
        
        # Organic keywords
        if any(word in prompt_lower for word in ['tree', 'plant', 'flower', 'organic',
                                                   'mountain', 'rock', 'crystal']):
            return "organic"
        
        return "generic"
    
    def _generate_vehicle(self, prompt: str, target_vertices: int, style: str) -> Mesh3D:
        """Generate vehicle-like geometry with wings and details"""
        segments = max(16, int(np.sqrt(target_vertices) / 2))
        
        vertices = []
        faces = []
        
        # Main fuselage body
        for i in range(segments):
            t = i / (segments - 1)
            
            # More aggressive taper for nose/tail
            if t < 0.3:  # Nose
                taper = t / 0.3
            elif t > 0.7:  # Tail
                taper = (1.0 - t) / 0.3
            else:  # Body
                taper = 1.0
            
            # Elliptical cross-section
            for j in range(segments):
                angle = 2 * np.pi * j / segments
                
                x = (t - 0.5) * 4  # Length (-2 to 2)
                y = np.cos(angle) * taper * 0.4  # Height
                z = np.sin(angle) * taper * 0.6  # Width (wider)
                
                vertices.append([x, y, z])
        
        fuselage_verts = len(vertices)
        
        # Add wings
        wing_segments = segments // 2
        for i in range(wing_segments):
            t = i / (wing_segments - 1)
            
            # Wing sweep
            x = -0.5 + t * 2  # Position along body
            y = -0.2  # Below center
            z_span = 1.5 + t * 0.5  # Wing span
            
            # Left wing
            vertices.append([x, y, -z_span])
            vertices.append([x, y - 0.1, -z_span * 0.9])
            
            # Right wing
            vertices.append([x, y, z_span])
            vertices.append([x, y - 0.1, z_span * 0.9])
        
        # Generate fuselage faces
        for i in range(segments - 1):
            for j in range(segments):
                j_next = (j + 1) % segments
                
                v1 = i * segments + j
                v2 = i * segments + j_next
                v3 = (i + 1) * segments + j
                v4 = (i + 1) * segments + j_next
                
                faces.extend([[v1, v2, v3], [v2, v4, v3]])
        
        # Generate wing faces
        wing_start = fuselage_verts
        for i in range(wing_segments - 1):
            base = wing_start + i * 4
            
            # Left wing
            faces.extend([
                [base, base + 1, base + 4],
                [base + 1, base + 5, base + 4]
            ])
            
            # Right wing
            faces.extend([
                [base + 2, base + 6, base + 3],
                [base + 3, base + 6, base + 7]
            ])
        
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.uint32).flatten()
        normals = self._compute_normals(vertices, faces)
        
        return Mesh3D(vertices=vertices, faces=faces, normals=normals)
    
    def _generate_character(self, prompt: str, target_vertices: int, style: str) -> Mesh3D:
        """Generate humanoid character geometry with limbs"""
        vertices = []
        faces = []
        
        segments = max(12, int(np.sqrt(target_vertices / 8)))
        
        def add_sphere(center, radius, segs):
            """Helper to add sphere geometry"""
            start_idx = len(vertices)
            for i in range(segs):
                theta = np.pi * i / (segs - 1)
                for j in range(segs):
                    phi = 2 * np.pi * j / segs
                    
                    x = center[0] + radius * np.sin(theta) * np.cos(phi)
                    y = center[1] + radius * np.cos(theta)
                    z = center[2] + radius * np.sin(theta) * np.sin(phi)
                    
                    vertices.append([x, y, z])
            
            # Add faces
            for i in range(segs - 1):
                for j in range(segs):
                    j_next = (j + 1) % segs
                    v1 = start_idx + i * segs + j
                    v2 = start_idx + i * segs + j_next
                    v3 = start_idx + (i + 1) * segs + j
                    v4 = start_idx + (i + 1) * segs + j_next
                    faces.extend([[v1, v2, v3], [v2, v4, v3]])
        
        def add_cylinder(start, end, radius, segs):
            """Helper to add cylinder geometry"""
            start_idx = len(vertices)
            direction = np.array(end) - np.array(start)
            length = np.linalg.norm(direction)
            
            for i in range(segs):
                t = i / (segs - 1)
                for j in range(segs):
                    angle = 2 * np.pi * j / segs
                    
                    x = start[0] + direction[0] * t + radius * np.cos(angle)
                    y = start[1] + direction[1] * t
                    z = start[2] + direction[2] * t + radius * np.sin(angle)
                    
                    vertices.append([x, y, z])
            
            # Add faces
            for i in range(segs - 1):
                for j in range(segs):
                    j_next = (j + 1) % segs
                    v1 = start_idx + i * segs + j
                    v2 = start_idx + i * segs + j_next
                    v3 = start_idx + (i + 1) * segs + j
                    v4 = start_idx + (i + 1) * segs + j_next
                    faces.extend([[v1, v2, v3], [v2, v4, v3]])
        
        # Head
        add_sphere([0, 1.4, 0], 0.25, segments)
        
        # Torso
        add_cylinder([0, 0.5, 0], [0, 1.1, 0], 0.3, segments)
        
        # Arms
        add_cylinder([0.3, 1.0, 0], [0.7, 0.5, 0], 0.1, segments // 2)  # Left upper
        add_cylinder([0.7, 0.5, 0], [0.8, 0.0, 0], 0.08, segments // 2)  # Left lower
        add_cylinder([-0.3, 1.0, 0], [-0.7, 0.5, 0], 0.1, segments // 2)  # Right upper
        add_cylinder([-0.7, 0.5, 0], [-0.8, 0.0, 0], 0.08, segments // 2)  # Right lower
        
        # Legs
        add_cylinder([0.15, 0.5, 0], [0.15, -0.2, 0], 0.12, segments // 2)  # Left upper
        add_cylinder([0.15, -0.2, 0], [0.15, -0.8, 0], 0.1, segments // 2)  # Left lower
        add_cylinder([-0.15, 0.5, 0], [-0.15, -0.2, 0], 0.12, segments // 2)  # Right upper
        add_cylinder([-0.15, -0.2, 0], [-0.15, -0.8, 0], 0.1, segments // 2)  # Right lower
        
        # Joints (shoulders, elbows, knees)
        add_sphere([0.3, 1.0, 0], 0.12, segments // 3)
        add_sphere([-0.3, 1.0, 0], 0.12, segments // 3)
        add_sphere([0.7, 0.5, 0], 0.1, segments // 3)
        add_sphere([-0.7, 0.5, 0], 0.1, segments // 3)
        add_sphere([0.15, -0.2, 0], 0.12, segments // 3)
        add_sphere([-0.15, -0.2, 0], 0.12, segments // 3)
        
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.uint32).flatten()
        normals = self._compute_normals(vertices, faces)
        
        return Mesh3D(vertices=vertices, faces=faces, normals=normals)
    
    def _generate_building(self, prompt: str, target_vertices: int, style: str) -> Mesh3D:
        """Generate building/architecture geometry with windows and details"""
        vertices = []
        faces = []
        
        # Create multi-story building
        floors = max(3, int(target_vertices / 300))
        sides = 4 if style == "low-poly" else 8
        
        for floor in range(floors + 1):
            # Taper building toward top
            floor_scale = 1.0 - (floor / (floors + 1)) * 0.2
            floor_height = floor * 0.5
            
            # Create floor vertices
            floor_start = len(vertices)
            for side in range(sides):
                angle = 2 * np.pi * side / sides
                x = floor_scale * np.cos(angle)
                z = floor_scale * np.sin(angle)
                
                vertices.append([x, floor_height, z])
        
        # Generate wall faces with window insets
        for floor in range(floors):
            floor_start = floor * sides
            
            for side in range(sides):
                side_next = (side + 1) % sides
                
                # Wall corners
                v1 = floor_start + side
                v2 = floor_start + sides + side
                v3 = floor_start + side_next
                v4 = floor_start + sides + side_next
                
                # Create wall with window inset
                # Outer wall frame
                faces.extend([[v1, v3, v2], [v2, v3, v4]])
                
                # Add window insets (create depth)
                if floor > 0:  # No windows on ground floor (door area)
                    # Window frame - slightly inset
                    window_depth = 0.05
                    
                    p1 = np.array(vertices[v1])
                    p2 = np.array(vertices[v2])
                    p3 = np.array(vertices[v3])
                    p4 = np.array(vertices[v4])
                    
                    # Calculate inset direction (toward center)
                    center = (p1 + p2 + p3 + p4) / 4
                    inset_dir = center / np.linalg.norm(center) * window_depth
                    
                    # Window vertices (inset)
                    w_start = len(vertices)
                    window_scale = 0.6
                    
                    # Interpolate for window position
                    w1 = p1 * (1 - window_scale) + (p1 + p2) / 2 * window_scale - inset_dir
                    w2 = p2 * (1 - window_scale) + (p1 + p2) / 2 * window_scale - inset_dir
                    w3 = p3 * (1 - window_scale) + (p3 + p4) / 2 * window_scale - inset_dir
                    w4 = p4 * (1 - window_scale) + (p3 + p4) / 2 * window_scale - inset_dir
                    
                    vertices.extend([w1.tolist(), w2.tolist(), w3.tolist(), w4.tolist()])
                    
                    # Window faces
                    faces.extend([
                        [w_start, w_start + 2, w_start + 1],
                        [w_start + 1, w_start + 2, w_start + 3]
                    ])
        
        # Add roof detail
        roof_height = floors * 0.5
        roof_peak = len(vertices)
        vertices.append([0, roof_height + 0.3, 0])
        
        # Roof faces
        top_floor_start = floors * sides
        for side in range(sides):
            side_next = (side + 1) % sides
            v1 = top_floor_start + side
            v2 = top_floor_start + side_next
            faces.extend([[v1, v2, roof_peak]])
        
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.uint32).flatten()
        normals = self._compute_normals(vertices, faces)
        
        return Mesh3D(vertices=vertices, faces=faces, normals=normals)
    
    def _generate_organic(self, prompt: str, target_vertices: int, style: str) -> Mesh3D:
        """Generate organic/natural geometry (trees, rocks, etc)"""
        # Start with icosahedron and subdivide with noise
        phi = (1 + np.sqrt(5)) / 2
        
        vertices = [
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ]
        
        faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ]
        
        # Subdivide to reach target complexity
        subdivisions = int(np.log2(target_vertices / 12))
        for _ in range(min(subdivisions, 4)):
            vertices, faces = self._subdivide_mesh(vertices, faces)
        
        vertices = np.array(vertices, dtype=np.float32)
        
        # Add organic noise
        noise_scale = 0.3
        for i in range(len(vertices)):
            # Use vertex position as seed for consistent noise
            seed = sum(vertices[i]) * 1000
            np.random.seed(int(seed) % 2**32)
            noise = np.random.randn(3) * noise_scale
            
            # Apply noise in normal direction
            normal = vertices[i] / (np.linalg.norm(vertices[i]) + 1e-6)
            vertices[i] = vertices[i] + normal * np.dot(noise, normal)
        
        faces = np.array(faces, dtype=np.uint32).flatten()
        normals = self._compute_normals(vertices, faces)
        
        return Mesh3D(vertices=vertices, faces=faces, normals=normals)
    
    def _generate_generic(self, prompt: str, target_vertices: int, style: str) -> Mesh3D:
        """Generate generic abstract geometry"""
        segments = int(np.sqrt(target_vertices))
        vertices = []
        faces = []
        
        # Create parametric surface based on prompt hash
        prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16)
        np.random.seed(prompt_hash % 2**32)
        
        freq1, freq2 = np.random.rand(2) * 3 + 1
        amp1, amp2 = np.random.rand(2) * 0.5 + 0.3
        
        for i in range(segments):
            u = i / (segments - 1) * 2 * np.pi
            for j in range(segments):
                v = j / (segments - 1) * 2 * np.pi
                
                x = np.cos(u) * (2 + np.cos(v) * amp1)
                y = np.sin(u) * (2 + np.cos(v) * amp1)
                z = np.sin(v) * amp2 + np.sin(freq1 * u) * 0.3
                
                vertices.append([x, y, z])
        
        # Generate faces
        for i in range(segments - 1):
            for j in range(segments - 1):
                v1 = i * segments + j
                v2 = i * segments + j + 1
                v3 = (i + 1) * segments + j
                v4 = (i + 1) * segments + j + 1
                
                faces.extend([[v1, v2, v3], [v2, v4, v3]])
        
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.uint32).flatten()
        normals = self._compute_normals(vertices, faces)
        
        return Mesh3D(vertices=vertices, faces=faces, normals=normals)
    
    def _subdivide_mesh(self, vertices: List, faces: List) -> Tuple[List, List]:
        """Subdivide mesh for higher detail"""
        new_vertices = list(vertices)
        new_faces = []
        midpoint_cache = {}
        
        def get_midpoint(v1: int, v2: int) -> int:
            key = tuple(sorted([v1, v2]))
            if key in midpoint_cache:
                return midpoint_cache[key]
            
            mid = [
                (vertices[v1][0] + vertices[v2][0]) / 2,
                (vertices[v1][1] + vertices[v2][1]) / 2,
                (vertices[v1][2] + vertices[v2][2]) / 2
            ]
            new_vertices.append(mid)
            idx = len(new_vertices) - 1
            midpoint_cache[key] = idx
            return idx
        
        for face in faces:
            v1, v2, v3 = face
            
            a = get_midpoint(v1, v2)
            b = get_midpoint(v2, v3)
            c = get_midpoint(v3, v1)
            
            new_faces.extend([
                [v1, a, c],
                [v2, b, a],
                [v3, c, b],
                [a, b, c]
            ])
        
        return new_vertices, new_faces
    
    def _compute_normals(self, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """Compute vertex normals"""
        normals = np.zeros_like(vertices)
        
        # Reshape faces to Nx3
        faces_reshaped = faces.reshape(-1, 3)
        
        for face in faces_reshaped:
            v1, v2, v3 = vertices[face]
            
            # Compute face normal
            edge1 = v2 - v1
            edge2 = v3 - v1
            face_normal = np.cross(edge1, edge2)
            
            # Accumulate to vertex normals
            normals[face[0]] += face_normal
            normals[face[1]] += face_normal
            normals[face[2]] += face_normal
        
        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normals = normals / norms
        
        return normals.astype(np.float32)


class ImageTo3DGenerator:
    """Generate 3D models from 2D images"""
    
    async def generate(self, image_data: bytes, depth_method: str = "automatic") -> Mesh3D:
        """Generate 3D mesh from image using depth estimation"""
        # Decode image (simplified - would use PIL/OpenCV in production)
        # For now, create heightmap-based mesh
        
        resolution = 64
        vertices = []
        faces = []
        
        # Create grid with height variation
        for i in range(resolution):
            for j in range(resolution):
                x = i / resolution - 0.5
                z = j / resolution - 0.5
                
                # Simulate depth from image (would use actual depth estimation)
                y = np.sin(x * 10) * np.cos(z * 10) * 0.1
                
                vertices.append([x, y, z])
        
        # Generate faces
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                v1 = i * resolution + j
                v2 = i * resolution + j + 1
                v3 = (i + 1) * resolution + j
                v4 = (i + 1) * resolution + j + 1
                
                faces.extend([[v1, v2, v3], [v2, v4, v3]])
        
        vertices = np.array(vertices, dtype=np.float32)
        faces = np.array(faces, dtype=np.uint32).flatten()
        
        # Compute normals
        generator = TextTo3DGenerator()
        normals = generator._compute_normals(vertices, faces)
        
        return Mesh3D(vertices=vertices, faces=faces, normals=normals)


class AITextureGenerator:
    """Generate textures for 3D models"""
    
    async def generate_texture(self, mesh: Mesh3D, style: str = "realistic",
                              resolution: int = 1024) -> bytes:
        """Generate texture for mesh"""
        # Simplified texture generation
        # Would use actual texture synthesis in production
        
        # Return placeholder texture data
        texture_size = resolution * resolution * 4  # RGBA
        texture_data = np.random.randint(0, 255, texture_size, dtype=np.uint8)
        
        return texture_data.tobytes()


def mesh_to_obj(mesh: Mesh3D) -> str:
    """Convert mesh to OBJ format"""
    obj_lines = ["# Generated by Gen3D Engine\n"]
    
    # Write vertices
    for v in mesh.vertices:
        obj_lines.append(f"v {v[0]} {v[1]} {v[2]}\n")
    
    # Write normals
    for n in mesh.normals:
        obj_lines.append(f"vn {n[0]} {n[1]} {n[2]}\n")
    
    # Write faces (OBJ indices start at 1)
    faces_reshaped = mesh.faces.reshape(-1, 3)
    for f in faces_reshaped:
        obj_lines.append(f"f {f[0]+1}//{f[0]+1} {f[1]+1}//{f[1]+1} {f[2]+1}//{f[2]+1}\n")
    
    return "".join(obj_lines)


def mesh_to_json(mesh: Mesh3D) -> dict:
    """Convert mesh to JSON format (Three.js compatible)"""
    return {
        "vertices": mesh.vertices.flatten().tolist(),
        "faces": mesh.faces.tolist(),
        "normals": mesh.normals.flatten().tolist(),
        "uvs": mesh.uvs.flatten().tolist() if mesh.uvs is not None else [],
        "colors": mesh.colors.flatten().tolist() if mesh.colors is not None else []
    }
