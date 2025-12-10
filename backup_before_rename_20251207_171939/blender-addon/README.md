# 🦅 Queztl-Core Blender Addon

**Connect your local Blender to Queztl-Core's WebGPU virtual driver!**

Test the driver's capabilities with real 3D workloads from Blender.

---

## 🚀 Quick Start

### 1. Install the Addon

1. **Open Blender** (3.0 or higher)
2. Go to: **Edit > Preferences > Add-ons**
3. Click **Install...**
4. Select: `blender-addon/queztl_gpu_addon.py`
5. ✅ Enable: **"Queztl-Core GPU Bridge"**

### 2. Start Queztl-Core Backend

```bash
cd /Users/xavasena/hive
./start.sh
```

Wait for:
```
✅ Queztl-Core backend running on http://localhost:8000
✅ Dashboard running on http://localhost:3000
```

### 3. Connect from Blender

1. In Blender, press **N** to open the sidebar
2. Select the **"Queztl GPU"** tab
3. Click **"Connect to Queztl GPU"**
4. See ✅ **"Connected to Queztl-Core!"** in the info bar

---

## 🎯 Features

### 1. **Test Render** - Offload Mesh Rendering
- Select any mesh object in Blender
- Click **"Test Render"** in the Queztl GPU panel
- Mesh data (vertices + triangles) is sent to the WebGPU driver
- Backend processes it and returns results

**What it does:**
- Extracts vertex positions and triangle indices
- Converts to GPU buffer format
- Submits render job via REST API
- Shows job ID and stats

### 2. **Run Benchmark** - Test GPU Performance
- Click **"Run Benchmark"**
- Executes the full Queztl-Core benchmark suite
- Shows:
  - Overall score (0-100)
  - Throughput (operations/second)
  - Latency (P95, P99)
  - Concurrency performance

**Use case:** Validate that your virtual GPU can handle real workloads

### 3. **Get Metrics** - Monitor Performance
- Click **"Get Metrics"**
- Retrieves last 10 performance metrics from database
- See historical performance data

### 4. **GPU Info** - View Capabilities
- Automatically fetched on connect
- Shows:
  - Number of cores
  - Global memory size
  - SIMD width
  - Compute capabilities

---

## 📋 Usage Examples

### Example 1: Test with Default Cube

1. Start Blender (default scene has a cube)
2. Connect to Queztl-Core
3. Select the cube (it should already be selected)
4. Click **"Test Render"**
5. Check the info bar: `✅ Render submitted! Job ID: ...`

**Output:**
```
Vertices: 8, Triangles: 12
✅ Render submitted! Job ID: abc123
```

### Example 2: Test with Complex Mesh

1. Add a UV Sphere: **Add > Mesh > UV Sphere**
2. Increase subdivisions: **F9** > Subdivisions: 5
3. Select the sphere
4. Click **"Test Render"**

**Output:**
```
Vertices: 2562, Triangles: 5120
✅ Render submitted!
```

### Example 3: Run Performance Benchmark

1. Connect to Queztl-Core
2. Click **"Run Benchmark"**
3. Wait ~10-15 seconds
4. See results:
   ```
   Score: 85.3/100
   Throughput: 5,820,000 ops/s
   ```

### Example 4: Monitor Over Time

1. Run several test renders
2. Click **"Get Metrics"**
3. See historical data (last 10 operations)
4. Open dashboard at http://localhost:3000 for full visualization

---

## 🔧 Configuration

### Change Server URL

If running Queztl-Core on a different machine:

1. In the Queztl GPU panel
2. Update **"Server URL"** field
3. Example: `http://192.168.1.100:8000`
4. Click **"Connect to Queztl GPU"**

### Adjust Render Resolution

1. In **"Test Render"** section
2. Change **Width** and **Height**
3. Default: 512x512
4. Max: 4096x4096

---

## 🛠️ Troubleshooting

### ❌ "Cannot connect. Is Queztl-Core running?"

**Solution:**
```bash
# Check if backend is running
curl http://localhost:8000/api/health

# If not, start it
cd /Users/xavasena/hive
./start.sh
```

### ❌ "Select a mesh object first"

**Solution:**
- Make sure you've selected a MESH object (not camera, light, etc.)
- In Object Mode, click on a mesh to select it

### ❌ "Render failed: error message"

**Solution:**
- Check backend logs: `docker-compose logs backend`
- Mesh might be too complex (>100K vertices)
- Try a simpler mesh first

### ❌ Addon not showing in preferences

**Solution:**
1. Check Blender version is 3.0+
2. Make sure you selected the `.py` file, not the folder
3. Try: Edit > Preferences > Add-ons > Search "Queztl"

---

## 📊 What Gets Tested

When you use the Blender addon, you're testing these WebGPU driver features:

### ✅ Buffer Operations
- **create_buffer**: Creates vertex and index buffers
- **write_buffer**: Uploads mesh data to GPU
- **read_buffer**: (future) Reads results back

### ✅ Data Transfer
- **REST API**: HTTP POST with JSON payload
- **Data serialization**: NumPy arrays → JSON → GPU buffers
- **Large data handling**: Meshes with thousands of vertices

### ✅ Render Pipeline
- **Vertex processing**: Transform vertices
- **Triangle assembly**: Index buffer to triangles
- **Rasterization**: (simulated) Convert to pixels

### ✅ Performance Validation
- **Real workloads**: Actual 3D mesh data, not synthetic
- **Benchmark comparison**: Compare Blender workload vs benchmark suite
- **Throughput**: Operations per second with real data

---

## 🎓 Understanding the Output

### Test Render Output

```json
{
  "job_id": "render_abc123",
  "status": "submitted",
  "vertices_count": 2562,
  "triangles_count": 5120,
  "estimated_time_ms": 45.2
}
```

**Meaning:**
- Job was accepted and queued
- Mesh has 2,562 vertices forming 5,120 triangles
- Expected to complete in ~45ms

### Benchmark Output

```json
{
  "overall_score": 85.3,
  "tests": {
    "throughput": {
      "operations_per_second": 5820000
    },
    "latency": {
      "p95_ms": 12.5
    }
  }
}
```

**Meaning:**
- System scored 85.3/100 (Grade: A - Excellent)
- Can handle 5.82M operations/second
- 95% of operations complete in <12.5ms

---

## 🔮 Future Features

### Coming Soon:
- [x] **Basic render submission** ✅ v1.1.0
- [ ] **Result visualization**: Display rendered image in Blender
- [ ] **Material support**: Send material/shader data
- [ ] **Animation**: Batch render animation frames
- [ ] **Progress tracking**: Real-time progress bar
- [ ] **WebSocket mode**: Live updates during rendering

### Advanced (Future):
- [ ] **Viewport integration**: Real-time viewport rendering
- [ ] **GPU acceleration selector**: Choose CPU/GPU/Queztl
- [ ] **Distributed rendering**: Multi-machine support
- [ ] **Shader compilation**: GLSL → Queztl compute kernels

---

## 📚 API Reference

The addon uses these Queztl-Core API endpoints:

### `GET /api/health`
Check if backend is running

### `GET /api/gpu/info`
Get GPU capabilities and configuration

### `POST /api/gpu/buffer/create`
Create a GPU buffer
```json
{
  "size": 4096,
  "buffer_type": "vertex",
  "usage": "dynamic"
}
```

### `POST /api/gpu/buffer/write`
Write data to buffer
```json
{
  "buffer_id": 0,
  "data": [1.0, 2.0, 3.0],
  "offset": 0
}
```

### `POST /api/gpu/render`
Submit render job
```json
{
  "vertices": [[x, y, z], ...],
  "indices": [0, 1, 2, ...],
  "width": 512,
  "height": 512
}
```

### `POST /api/power/benchmark`
Run performance benchmark

### `GET /api/metrics`
Get performance metrics history

---

## 💡 Tips & Best Practices

### For Best Results:

1. **Start Simple**: Test with default cube before complex models
2. **Gradual Complexity**: Cube (8 verts) → Sphere (482 verts) → Suzanne (507 verts) → Complex model
3. **Monitor Performance**: Run benchmark before/after to compare
4. **Check Logs**: Keep terminal open to see backend processing
5. **Use Dashboard**: Open http://localhost:3000 for visual monitoring

### Performance Expectations:

| Mesh Complexity | Vertices | Expected Time |
|----------------|----------|---------------|
| Cube           | 8        | <1ms          |
| UV Sphere      | 482      | ~5ms          |
| Suzanne        | 507      | ~5ms          |
| Subdivided (2) | 2K       | ~20ms         |
| Subdivided (3) | 8K       | ~80ms         |
| Complex Model  | 50K      | ~500ms        |

---

## 🤝 Contributing

Found a bug or have a feature request?

1. Check existing issues on GitHub
2. Create a new issue with:
   - Blender version
   - Addon version
   - Steps to reproduce
   - Expected vs actual behavior

---

## 📄 License

Copyright (c) 2025 Queztl-Core Project  
See LICENSE file for details

---

## 🎉 Success!

You now have Blender connected to your Queztl-Core virtual GPU! 🚀

**What you can do:**
- ✅ Test WebGPU driver with real 3D data
- ✅ Benchmark performance with actual workloads
- ✅ Validate buffer operations
- ✅ Monitor GPU metrics in real-time

**Next Steps:**
1. Try different mesh complexities
2. Run benchmarks to validate performance
3. Check the dashboard for visualizations
4. Prepare for pen testing with real data

Happy rendering! 🦅✨
