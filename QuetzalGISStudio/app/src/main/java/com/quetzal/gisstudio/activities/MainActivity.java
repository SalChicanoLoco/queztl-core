package com.quetzal.gisstudio.activities;

import androidx.appcompat.app.AppCompatActivity;
import androidx.viewpager2.widget.ViewPager2;
import android.os.Bundle;
import android.view.View;
import com.google.android.material.bottomnavigation.BottomNavigationView;
import com.google.android.material.tabs.TabLayout;
import com.google.android.material.tabs.TabLayoutMediator;
import com.quetzal.gisstudio.R;
import com.quetzal.gisstudio.adapters.MainPagerAdapter;
import timber.log.Timber;

public class MainActivity extends AppCompatActivity {

    private ViewPager2 viewPager;
    private BottomNavigationView navView;
    private MainPagerAdapter pagerAdapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        Timber.d("🗺️ Quetzal GIS Studio Initializing...");
        
        initializeUI();
        setupNavigation();
        loadInitialData();
    }

    private void initializeUI() {
        viewPager = findViewById(R.id.view_pager);
        navView = findViewById(R.id.bottom_nav);
        
        pagerAdapter = new MainPagerAdapter(this);
        viewPager.setAdapter(pagerAdapter);
        
        viewPager.registerOnPageChangeCallback(new ViewPager2.OnPageChangeCallback() {
            @Override
            public void onPageSelected(int position) {
                navView.getMenu().getItem(position).setChecked(true);
            }
        });
    }

    private void setupNavigation() {
        navView.setOnItemSelectedListener(item -> {
            int itemId = item.getItemId();
            if (itemId == R.id.nav_map) {
                viewPager.setCurrentItem(0);
            } else if (itemId == R.id.nav_dashboard) {
                viewPager.setCurrentItem(1);
            } else if (itemId == R.id.nav_analysis) {
                viewPager.setCurrentItem(2);
            } else if (itemId == R.id.nav_settings) {
                viewPager.setCurrentItem(3);
            }
            return true;
        });
    }

    private void loadInitialData() {
        Timber.d("Loading Quetzal GIS data...");
        // Initialize GIS modules
        // Connect to backend
        // Load offline tiles
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        Timber.d("Quetzal GIS Studio closing");
    }
}
