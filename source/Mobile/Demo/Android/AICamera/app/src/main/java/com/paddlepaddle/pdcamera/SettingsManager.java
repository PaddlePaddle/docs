package com.paddlepaddle.pdcamera;

import android.content.Context;
import android.content.SharedPreferences;

/**
 * Created by Nickychan on 1/17/18.
 */

public class SettingsManager {

    interface SettingsListener {
        void onCameraChanged(boolean backCamera);
        void onModelChanged(SSDModel ssdModel);
        void onThresholdChanged(float threshold);
        void onSettingsClose();
    }

    private static final String PREF_NAME = "prefs";
    private static final String PREF_KEY_THRESHOLD = "accuracy_threshold";
    private static final String PREF_KEY_MODEL = "ssd_model";
    private static final String PREF_KEY_CAMERA = "camera";

    private static SettingsManager sInstance = new SettingsManager();

    private SettingsListener mListener;

    private boolean mBackCamera;
    private SSDModel mModel;
    private float mThreshold;

    private SharedPreferences mPrefs;

    public static SettingsManager getInstance() {
        return sInstance;
    }

    public void setListener(SettingsListener listener) {
        mListener = listener;
    }

    public void loadSettings(Context context) {
        mPrefs = context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE);
        mBackCamera = mPrefs.getBoolean(PREF_KEY_CAMERA, true);
        mThreshold = mPrefs.getFloat(PREF_KEY_THRESHOLD, 0.3f);
        String modelFileName = mPrefs.getString(PREF_KEY_MODEL, SSDModel.PASCAL_MOBILENET_300.modelFileName);
        mModel = SSDModel.fromModelFileName(modelFileName);
    }

    public void cameraChanged(boolean backCamera) {
        mBackCamera = backCamera;
        SharedPreferences.Editor editor = mPrefs.edit();
        editor.putBoolean(PREF_KEY_CAMERA, backCamera);
        editor.commit();
        if (mListener != null) mListener.onCameraChanged(backCamera);
    }

    public void modelChanged(SSDModel ssdModel) {
        mModel = ssdModel;
        SharedPreferences.Editor editor = mPrefs.edit();
        editor.putString(PREF_KEY_MODEL, ssdModel.modelFileName);
        editor.commit();
        if (mListener != null) mListener.onModelChanged(ssdModel);
    }

    public void thresholdChanged(float threshold) {
        mThreshold = threshold;
        SharedPreferences.Editor editor = mPrefs.edit();
        editor.putFloat(PREF_KEY_THRESHOLD, threshold);
        editor.commit();
        if (mListener != null) mListener.onThresholdChanged(threshold);
    }

    public void closeSettings() {
        if (mListener != null) mListener.onSettingsClose();
    }

    public boolean isBackCamera() {
        return mBackCamera;
    }

    public SSDModel getModel() {
        return mModel;
    }

    public float getThreshold() {
        return mThreshold;
    }

}
