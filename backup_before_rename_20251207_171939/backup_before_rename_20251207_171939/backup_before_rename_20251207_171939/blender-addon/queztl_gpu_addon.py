"""
🦅 QUEZTL-CORE BLENDER ADDON
Connect Blender to Queztl-Core Virtual GPU

This addon enables Blender to:
- Offload rendering to Queztl-Core software GPU
- Test WebGPU driver capabilities
- Benchmark GPU operations with real 3D workloads
- Monitor performance in real-time

Installation:
1. Open Blender
2. Edit > Preferences > Add-ons > Install
3. Select this file
4. Enable "Queztl-Core GPU Bridge"

Usage:
1. Start Queztl-Core backend: ./start.sh
2. In Blender: View3D > Sidebar (N) > Queztl GPU
3. Click "Connect" to establish connection
4. Use "Test Render" to offload a frame
"""

bl_info = {
    "name": "Queztl-Core GPU Bridge",
    "author": "Queztl-Core Team",
    "version": (1, 1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Queztl GPU",
    "description": "Connect Blender to Queztl-Core virtual GPU for offloading rendering",
    "warning": "Requires Queztl-Core backend running on localhost:8000",
    "doc_url": "https://github.com/quetzalcore-core/docs",
    "category": "Render",
}

import bpy
import requests
import json
import numpy as np
from bpy.types import Panel, Operator, PropertyGroup
from bpy.props import StringProperty, BoolProperty, IntProperty, EnumProperty


# ============================================================================
# QUEZTL-CORE API CLIENT
# ============================================================================

class QueztlGPUClient:
    """Client for Queztl-Core WebGPU API"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session_id = None
        self.connected = False
    
    def check_connection(self):
        """Check if Queztl-Core is running"""
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def get_gpu_info(self):
        """Get GPU capabilities"""
        try:
            response = requests.get(f"{self.base_url}/api/gpu/info")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def create_buffer(self, size, buffer_type="vertex"):
        """Create GPU buffer"""
        data = {
            "size": size,
            "buffer_type": buffer_type,
            "usage": "dynamic"
        }
        response = requests.post(f"{self.base_url}/api/gpu/buffer/create", json=data)
        return response.json()
    
    def write_buffer(self, buffer_id, data):
        """Write data to GPU buffer"""
        payload = {
            "buffer_id": buffer_id,
            "data": data.tolist() if isinstance(data, np.ndarray) else data,
            "offset": 0
        }
        response = requests.post(f"{self.base_url}/api/gpu/buffer/write", json=payload)
        return response.json()
    
    def submit_render_job(self, vertices, indices, width=512, height=512):
        """Submit rendering job to GPU"""
        job_data = {
            "vertices": vertices.tolist() if isinstance(vertices, np.ndarray) else vertices,
            "indices": indices.tolist() if isinstance(indices, np.ndarray) else indices,
            "width": width,
            "height": height
        }
        response = requests.post(f"{self.base_url}/api/gpu/render", json=job_data)
        return response.json()
    
    def run_benchmark(self):
        """Run GPU benchmark"""
        response = requests.post(f"{self.base_url}/api/power/benchmark")
        return response.json()
    
    def get_metrics(self):
        """Get performance metrics"""
        response = requests.get(f"{self.base_url}/api/metrics")
        return response.json()


# ============================================================================
# MESH EXPORT UTILITIES
# ============================================================================

def extract_mesh_data(obj):
    """Extract vertices and faces from Blender mesh"""
    if obj.type != 'MESH':
        return None, None
    
    mesh = obj.data
    
    # Get vertices (convert to numpy)
    vertices = np.zeros((len(mesh.vertices) * 3,), dtype=np.float32)
    mesh.vertices.foreach_get('co', vertices)
    vertices = vertices.reshape(-1, 3)
    
    # Get faces as triangles
    mesh.calc_loop_triangles()
    indices = np.zeros((len(mesh.loop_triangles) * 3,), dtype=np.int32)
    mesh.loop_triangles.foreach_get('vertices', indices)
    
    return vertices, indices


def mesh_to_gpu_format(vertices, indices):
    """Convert mesh data to GPU-compatible format"""
    # Pack vertices with position + normal + uv
    # For now, just position (3 floats per vertex)
    vertex_data = vertices.flatten()
    index_data = indices.flatten()
    
    return vertex_data, index_data


# ============================================================================
# OPERATORS
# ============================================================================

class QUEZTL_OT_Connect(Operator):
    """Connect to Queztl-Core GPU"""
    bl_idname = "quetzalcore.connect"
    bl_label = "Connect to Queztl GPU"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        props = context.scene.quetzalcore_props
        client = QueztlGPUClient(props.server_url)
        
        if client.check_connection():
            props.is_connected = True
            self.report({'INFO'}, "✅ Connected to Queztl-Core!")
            
            # Get GPU info
            info = client.get_gpu_info()
            if "error" not in info:
                props.gpu_info = json.dumps(info, indent=2)
        else:
            props.is_connected = False
            self.report({'ERROR'}, "❌ Cannot connect. Is Queztl-Core running?")
        
        return {'FINISHED'}


class QUEZTL_OT_Disconnect(Operator):
    """Disconnect from Queztl-Core GPU"""
    bl_idname = "quetzalcore.disconnect"
    bl_label = "Disconnect"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        props = context.scene.quetzalcore_props
        props.is_connected = False
        self.report({'INFO'}, "Disconnected from Queztl-Core")
        return {'FINISHED'}


class QUEZTL_OT_TestRender(Operator):
    """Test render with selected object"""
    bl_idname = "quetzalcore.test_render"
    bl_label = "Test Render"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        props = context.scene.quetzalcore_props
        
        if not props.is_connected:
            self.report({'ERROR'}, "Not connected to Queztl-Core")
            return {'CANCELLED'}
        
        # Get active object
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "Select a mesh object first")
            return {'CANCELLED'}
        
        # Extract mesh data
        self.report({'INFO'}, f"Extracting mesh data from '{obj.name}'...")
        vertices, indices = extract_mesh_data(obj)
        
        if vertices is None:
            self.report({'ERROR'}, "Failed to extract mesh data")
            return {'CANCELLED'}
        
        self.report({'INFO'}, f"Vertices: {len(vertices)}, Triangles: {len(indices)//3}")
        
        # Submit to GPU
        client = QueztlGPUClient(props.server_url)
        
        try:
            result = client.submit_render_job(
                vertices, 
                indices,
                width=props.render_width,
                height=props.render_height
            )
            
            if "error" in result:
                self.report({'ERROR'}, f"Render failed: {result['error']}")
            else:
                self.report({'INFO'}, f"✅ Render submitted! Job ID: {result.get('job_id', 'N/A')}")
                props.last_render_stats = json.dumps(result, indent=2)
        
        except Exception as e:
            self.report({'ERROR'}, f"Error: {str(e)}")
            return {'CANCELLED'}
        
        return {'FINISHED'}


class QUEZTL_OT_Benchmark(Operator):
    """Run GPU benchmark"""
    bl_idname = "quetzalcore.benchmark"
    bl_label = "Run Benchmark"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        props = context.scene.quetzalcore_props
        
        if not props.is_connected:
            self.report({'ERROR'}, "Not connected to Queztl-Core")
            return {'CANCELLED'}
        
        client = QueztlGPUClient(props.server_url)
        
        try:
            self.report({'INFO'}, "Running benchmark...")
            result = client.run_benchmark()
            
            if "overall_score" in result:
                score = result['overall_score']
                throughput = result['tests']['throughput']['operations_per_second']
                self.report({'INFO'}, f"✅ Score: {score:.1f}/100, Throughput: {throughput:,.0f} ops/sec")
                props.last_benchmark_stats = json.dumps(result, indent=2)
            else:
                self.report({'ERROR'}, "Benchmark failed")
        
        except Exception as e:
            self.report({'ERROR'}, f"Error: {str(e)}")
            return {'CANCELLED'}
        
        return {'FINISHED'}


class QUEZTL_OT_GetMetrics(Operator):
    """Get performance metrics"""
    bl_idname = "quetzalcore.get_metrics"
    bl_label = "Get Metrics"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        props = context.scene.quetzalcore_props
        
        if not props.is_connected:
            self.report({'ERROR'}, "Not connected to Queztl-Core")
            return {'CANCELLED'}
        
        client = QueztlGPUClient(props.server_url)
        
        try:
            metrics = client.get_metrics()
            self.report({'INFO'}, f"✅ Retrieved {len(metrics)} metrics")
            props.metrics_data = json.dumps(metrics[:10], indent=2)  # Show last 10
        
        except Exception as e:
            self.report({'ERROR'}, f"Error: {str(e)}")
            return {'CANCELLED'}
        
        return {'FINISHED'}


# ============================================================================
# PROPERTIES
# ============================================================================

class QueztlProperties(PropertyGroup):
    """Addon properties"""
    
    server_url: StringProperty(
        name="Server URL",
        description="Queztl-Core API endpoint",
        default="http://localhost:8000"
    )
    
    is_connected: BoolProperty(
        name="Connected",
        description="Connection status",
        default=False
    )
    
    render_width: IntProperty(
        name="Width",
        description="Render width",
        default=512,
        min=64,
        max=4096
    )
    
    render_height: IntProperty(
        name="Height",
        description="Render height",
        default=512,
        min=64,
        max=4096
    )
    
    gpu_info: StringProperty(
        name="GPU Info",
        description="GPU capabilities",
        default=""
    )
    
    last_render_stats: StringProperty(
        name="Render Stats",
        description="Last render statistics",
        default=""
    )
    
    last_benchmark_stats: StringProperty(
        name="Benchmark Stats",
        description="Last benchmark results",
        default=""
    )
    
    metrics_data: StringProperty(
        name="Metrics",
        description="Performance metrics",
        default=""
    )


# ============================================================================
# UI PANEL
# ============================================================================

class QUEZTL_PT_MainPanel(Panel):
    """Queztl-Core GPU Bridge Panel"""
    bl_label = "Queztl GPU Bridge"
    bl_idname = "QUEZTL_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Queztl GPU'
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.quetzalcore_props
        
        # Connection Section
        box = layout.box()
        box.label(text="Connection", icon='NETWORK_DRIVE')
        box.prop(props, "server_url")
        
        row = box.row()
        if props.is_connected:
            row.label(text="Status: Connected ✅", icon='CHECKMARK')
            row = box.row()
            row.operator("quetzalcore.disconnect", icon='CANCEL')
        else:
            row.label(text="Status: Disconnected ❌", icon='ERROR')
            row = box.row()
            row.operator("quetzalcore.connect", icon='PLUGIN')
        
        layout.separator()
        
        # GPU Info Section
        if props.is_connected and props.gpu_info:
            box = layout.box()
            box.label(text="GPU Info", icon='INFO')
            
            try:
                info = json.loads(props.gpu_info)
                box.label(text=f"Cores: {info.get('num_cores', 'N/A')}")
                box.label(text=f"Memory: {info.get('global_memory_size', 'N/A')} bytes")
                box.label(text=f"SIMD Width: {info.get('simd_width', 'N/A')}")
            except:
                box.label(text="Unable to parse GPU info")
        
        layout.separator()
        
        # Render Section
        if props.is_connected:
            box = layout.box()
            box.label(text="Test Render", icon='RENDER_STILL')
            box.prop(props, "render_width")
            box.prop(props, "render_height")
            box.operator("quetzalcore.test_render", icon='PLAY')
            
            if props.last_render_stats:
                box.label(text="Last Render:")
                try:
                    stats = json.loads(props.last_render_stats)
                    if "job_id" in stats:
                        box.label(text=f"Job ID: {stats['job_id']}")
                except:
                    pass
        
        layout.separator()
        
        # Benchmark Section
        if props.is_connected:
            box = layout.box()
            box.label(text="Benchmark", icon='EXPERIMENTAL')
            box.operator("quetzalcore.benchmark", icon='TIME')
            
            if props.last_benchmark_stats:
                try:
                    stats = json.loads(props.last_benchmark_stats)
                    if "overall_score" in stats:
                        box.label(text=f"Score: {stats['overall_score']:.1f}/100")
                        throughput = stats['tests']['throughput']['operations_per_second']
                        box.label(text=f"Throughput: {throughput:,.0f} ops/s")
                except:
                    pass
        
        layout.separator()
        
        # Metrics Section
        if props.is_connected:
            box = layout.box()
            box.label(text="Metrics", icon='GRAPH')
            box.operator("quetzalcore.get_metrics", icon='FILE_REFRESH')


# ============================================================================
# REGISTRATION
# ============================================================================

classes = (
    QueztlProperties,
    QUEZTL_OT_Connect,
    QUEZTL_OT_Disconnect,
    QUEZTL_OT_TestRender,
    QUEZTL_OT_Benchmark,
    QUEZTL_OT_GetMetrics,
    QUEZTL_PT_MainPanel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.quetzalcore_props = bpy.props.PointerProperty(type=QueztlProperties)
    
    print("✅ Queztl-Core GPU Bridge addon registered!")

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    del bpy.types.Scene.quetzalcore_props
    
    print("Queztl-Core GPU Bridge addon unregistered")

if __name__ == "__main__":
    register()
