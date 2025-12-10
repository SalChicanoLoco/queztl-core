# Quetzal GIS Studio - Android App Build Guide

## 📱 Project Overview

This is a complete Android application for Quetzal GIS Studio with:
- Offline-first architecture
- Real-time GIS analysis
- Terrain visualization
- Data validation
- Multi-source data fusion
- WebSocket connectivity to Quetzal backend

## 🚀 Quick Start

### Prerequisites
- Android Studio 2023.1.1 or later
- JDK 11 or later
- Android SDK 24+ (API level)
- Gradle 8.1+

### Build Steps

1. **Open in Android Studio:**
   ```bash
   cd QuetzalGISStudio
   # Open in Android Studio
   ```

2. **Sync Gradle:**
   - Click "File" → "Sync Now"
   - Wait for dependencies to download

3. **Build APK:**
   ```bash
   ./gradlew assembleDebug      # Debug build
   ./gradlew assembleRelease    # Release build
   ```

4. **Run on Emulator:**
   - Create AVD (Android Virtual Device)
   - Select device and start
   - Click "Run" → "Run 'app'"

5. **Deploy to Device:**
   ```bash
   adb install app/build/outputs/apk/debug/app-debug.apk
   ```

## 📁 Project Structure

```
QuetzalGISStudio/
├── app/
│   ├── src/
│   │   ├── main/
│   │   │   ├── java/com/quetzal/gisstudio/
│   │   │   │   ├── activities/     # UI screens
│   │   │   │   ├── fragments/      # UI fragments
│   │   │   │   ├── services/       # Background services
│   │   │   │   ├── models/         # Data models
│   │   │   │   └── utils/          # Utility classes
│   │   │   ├── res/               # Resources
│   │   │   │   ├── layout/        # XML layouts
│   │   │   │   ├── values/        # Strings, colors, styles
│   │   │   │   ├── drawable/      # Images & vectors
│   │   │   │   └── menu/          # Menu layouts
│   │   │   └── AndroidManifest.xml
│   │   ├── test/                  # Unit tests
│   │   └── androidTest/           # Instrumentation tests
│   ├── build.gradle               # App dependencies
│   └── proguard-rules.pro         # Code obfuscation
├── build.gradle                   # Project config
├── settings.gradle                # Module settings
└── gradle.properties              # Gradle properties
```

## 🔌 Backend Connection

The app connects to Quetzal backend at:
```
ws://10.168.222.67:8000/ws  (WebSocket)
http://10.168.222.67:8000   (REST API)
```

### Configure Endpoint

Edit `app/src/main/java/com/quetzal/gisstudio/utils/ApiClient.java`:

```java
public class ApiClient {
    private static final String BASE_URL = "http://10.168.222.67:8000/";
    private static final String WS_URL = "ws://10.168.222.67:8000/ws";
    
    // Update these for different environments
}
```

## 📦 Key Dependencies

- **Mapping:** Mapsforge (offline map rendering)
- **Networking:** Retrofit 2 + OkHttp
- **Database:** Room (offline data)
- **Charts:** MPAndroidChart
- **Location:** Google Play Services
- **GIS:** GDAL bindings (optional)

## 🏗️ Building for Production

### 1. Create Keystore

```bash
keytool -genkey -v -keystore quetzal.keystore   -keyalg RSA -keysize 2048 -validity 10000   -alias quetzal-key
```

### 2. Configure Signing

Edit `app/build.gradle`:

```gradle
signingConfigs {
    release {
        storeFile file('quetzal.keystore')
        storePassword 'YOUR_STORE_PASSWORD'
        keyAlias 'quetzal-key'
        keyPassword 'YOUR_KEY_PASSWORD'
    }
}
```

### 3. Build Release APK

```bash
./gradlew assembleRelease --info
```

Output: `app/build/outputs/apk/release/app-release.apk`

### 4. Build AAB for Play Store

```bash
./gradlew bundleRelease --info
```

Output: `app/build/outputs/bundle/release/app-release.aab`

## 🧪 Testing

### Unit Tests
```bash
./gradlew test
```

### Instrumentation Tests
```bash
./gradlew connectedAndroidTest
```

### Run Specific Test
```bash
./gradlew test --tests com.quetzal.gisstudio.GISEngineTest
```

## 🐛 Debugging

### Enable Debugging
```bash
adb shell setprop debug.atrace.tags.enableflags 1
```

### View Logs
```bash
adb logcat | grep "GIS"
```

### Debug Session
- Set breakpoint in Android Studio
- Run in debug mode: Shift+F9
- Use debugger panel

## ⚙️ Features

### Current
- ✅ Interactive map view
- ✅ Dashboard with metrics
- ✅ Terrain analysis
- ✅ Data validation (LiDAR, Raster, Vector)
- ✅ Multi-source fusion
- ✅ Offline caching
- ✅ WebSocket sync

### Roadmap
- 🔲 AR terrain visualization
- 🔲 Real-time GPS tracking
- 🔲 Advanced GIS tools
- 🔲 Custom map layers
- 🔲 Data export (GeoJSON, ShapeFile)

## 🔒 Security

- ✅ SSL/TLS for all connections
- ✅ Encrypted local database
- ✅ Permission management
- ✅ Secure credential storage
- ✅ Code obfuscation (ProGuard)

## 📱 Supported Devices

- **Minimum API:** 24 (Android 7.0)
- **Target API:** 34 (Android 14)
- **Screen Sizes:** Phone & Tablet (4.5" to 7")
- **Architectures:** arm64-v8a, armeabi-v7a

## 🚀 Deployment

### Google Play Store
1. Create Play Store account
2. Build release AAB
3. Upload to Play Console
4. Configure listing & pricing
5. Submit for review

### Direct APK Distribution
```bash
# Build APK
./gradlew assembleRelease

# Sign APK
jarsigner -verbose -sigalg SHA1withRSA -digestalg SHA1   app-release.apk quetzal.keystore

# Verify
jarsigner -verify -verbose app-release.apk
```

## 📖 Documentation

- Android Docs: https://developer.android.com
- Retrofit: https://square.github.io/retrofit/
- Room: https://developer.android.com/training/data-storage/room
- Mapsforge: https://github.com/mapsforge/mapsforge

## 🆘 Troubleshooting

### Build Issues
```bash
./gradlew clean
./gradlew build --stacktrace
```

### Dependency Conflicts
```bash
./gradlew app:dependencies
```

### Runtime Issues
Check logcat for detailed error messages

## 📞 Support

For issues with:
- **Backend:** Check Quetzal API is running on port 8000
- **Maps:** Verify offline map files are present
- **Location:** Ensure permissions are granted
- **Network:** Check firewall rules

---

**Ready to build? Start with:**
```bash
cd QuetzalGISStudio
./gradlew assembleDebug
```

🗺️ Happy mapping! 🚀
