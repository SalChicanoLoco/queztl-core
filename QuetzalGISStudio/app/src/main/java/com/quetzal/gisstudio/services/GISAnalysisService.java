package com.quetzal.gisstudio.services;

import android.app.Service;
import android.content.Intent;
import android.os.Binder;
import android.os.IBinder;
import androidx.annotation.Nullable;
import com.quetzal.gisstudio.models.GISAnalysis;
import com.quetzal.gisstudio.utils.GISEngine;
import timber.log.Timber;
import java.util.ArrayList;
import java.util.List;

public class GISAnalysisService extends Service {

    private final IBinder binder = new LocalBinder();
    private GISEngine gisEngine;
    private List<GISAnalysis> analysisHistory;

    public class LocalBinder extends Binder {
        public GISAnalysisService getService() {
            return GISAnalysisService.this;
        }
    }

    @Override
    public void onCreate() {
        super.onCreate();
        gisEngine = new GISEngine();
        analysisHistory = new ArrayList<>();
        Timber.d("🌍 GIS Analysis Service started");
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        Timber.d("Processing GIS analysis task");
        return START_STICKY;
    }

    public GISAnalysis performTerrainAnalysis(double latitude, double longitude) {
        Timber.d("Analyzing terrain at %.4f, %.4f", latitude, longitude);
        return gisEngine.analyzeTerrainAtLocation(latitude, longitude);
    }

    public GISAnalysis performValidation(String dataType, String filePath) {
        Timber.d("Validating %s data: %s", dataType, filePath);
        return gisEngine.validateGISData(dataType, filePath);
    }

    public GISAnalysis performMultimodalFusion(List<String> dataSources) {
        Timber.d("Fusing %d data sources", dataSources.size());
        return gisEngine.fuseMultimodalData(dataSources);
    }

    public List<GISAnalysis> getAnalysisHistory() {
        return analysisHistory;
    }

    public void clearHistory() {
        analysisHistory.clear();
        Timber.d("Cleared analysis history");
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return binder;
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        Timber.d("GIS Analysis Service destroyed");
    }
}
