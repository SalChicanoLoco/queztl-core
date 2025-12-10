# 🗺️ Quetzal GIS Studio - Android App

A powerful, offline-first Android application for geospatial analysis powered by Quetzal Core.

## ✨ Features

### 📍 Mapping & Navigation
- Interactive offline map view with Mapsforge
- Real-time GPS positioning
- Custom layer support
- Zoom to location

### 🌍 GIS Analysis
- **Terrain Analysis:** Elevation, slope, aspect
- **Data Validation:** LiDAR, Raster, Vector
- **Multi-modal Fusion:** Combine multiple data sources
- **Geophysics Integration:** Gravity, magnetic fields

### 📊 Dashboard & Visualization
- Real-time metrics
- Analysis history
- Interactive charts
- Performance monitoring

### 💾 Offline Capabilities
- Local data caching
- Offline map tiles
- SQLite database
- Sync when online

### 🔗 Backend Integration
- WebSocket connection to Quetzal
- QP Protocol (binary, 10-20x faster)
- REST API fallback
- Real-time updates

## 📥 Quick Start

### Build
```bash
./gradlew assembleDebug
```

### Run
```bash
adb install app/build/outputs/apk/debug/app-debug.apk
```

### Debug
```bash
./gradlew connectedAndroidTest
```

## 📋 System Requirements

- Android 7.0+ (API 24)
- 2GB RAM minimum
- 100MB storage
- Network for sync

## 🏗️ Architecture

```
┌─────────────────────────────────────┐
│   Android UI (Activities/Fragments)  │
├─────────────────────────────────────┤
│   Services (GIS, Sync, Location)    │
├─────────────────────────────────────┤
│   Models & Database (Room/SQLite)   │
├─────────────────────────────────────┤
│   Utils (GIS Engine, Maps, API)     │
├─────────────────────────────────────┤
│   Quetzal Backend (WebSocket)       │
└─────────────────────────────────────┘
```

## 🔌 Backend Configuration

Default: `http://10.168.222.67:8000`

Update in: `ApiClient.java`

```java
BASE_URL = "http://YOUR_IP:8000/"
WS_URL = "ws://YOUR_IP:8000/ws"
```

## 📦 Dependencies

- Retrofit 2 - HTTP client
- Room - Local database
- Mapsforge - Offline maps
- Timber - Logging
- Glide - Image loading

See `app/build.gradle` for complete list.

## 🧪 Testing

```bash
# Unit tests
./gradlew test

# Instrumentation tests
./gradlew connectedAndroidTest

# Coverage report
./gradlew testDebugCoverage
```

## 📱 Supported Devices

- Phones: 4.5" - 6.5" (common)
- Tablets: 7" - 12"
- Orientations: Portrait & Landscape
- Min API: 24 (Android 7.0)
- Target API: 34 (Android 14)

## 🚀 Release Build

```bash
# Create keystore
keytool -genkey -v -keystore quetzal.keystore   -keyalg RSA -keysize 2048 -validity 10000

# Build release
./gradlew assembleRelease

# Output: app/build/outputs/apk/release/app-release.apk
```

## 🔒 Security

- ✅ SSL/TLS encryption
- ✅ Encrypted database
- ✅ Permission management
- ✅ Code obfuscation
- ✅ Secure credentials

## 📚 Documentation

- [Android Build Guide](ANDROID_BUILD_GUIDE.md)
- [API Documentation](../UBUNTU_DEPLOYMENT_SUMMARY.txt)
- [Quetzal Docs](../README.md)

## 🆘 Troubleshooting

**App crashes on startup?**
- Check backend is running
- Verify API endpoint
- Check logcat: `adb logcat`

**Map not loading?**
- Ensure offline maps are present
- Check storage permissions
- Verify tiles format

**Location not working?**
- Grant location permission
- Enable location in settings
- Check GPS is turned on

## 🤝 Contributing

Pull requests welcome! Please:
1. Follow Android conventions
2. Add tests for new features
3. Update documentation
4. Test on multiple devices

## 📄 License

Built with ❤️ for Quetzal Core

---

**Ready to map the world?** 🗺️🚀

Start building:
```bash
./gradlew assembleDebug
```
