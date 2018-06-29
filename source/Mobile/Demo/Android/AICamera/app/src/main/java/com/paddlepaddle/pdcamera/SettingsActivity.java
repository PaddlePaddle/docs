package com.paddlepaddle.pdcamera;

import android.app.Activity;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.view.View;
import android.widget.CompoundButton;
import android.widget.RadioButton;
import android.widget.SeekBar;
import android.widget.TextView;

import butterknife.BindView;
import butterknife.ButterKnife;
import butterknife.OnCheckedChanged;
import butterknife.OnClick;

/**
 * Created by Nickychan on 1/17/18.
 */

public class SettingsActivity extends Activity{

    @BindView(R.id.pascalRadioBtn)
    RadioButton mPascalRadioBtn;

    @BindView(R.id.faceRadioBtn)
    RadioButton mFaceRadioBtn;

    @BindView(R.id.backCameraRadioBtn)
    RadioButton mBackCameraRadioBtn;

    @BindView(R.id.frontCameraRadioBtn)
    RadioButton mFrontCameraRadioBtn;

    @BindView(R.id.accuracySlider)
    SeekBar mAccuracySlider;

    @BindView(R.id.accuracyText)
    TextView mAccuracyText;

    @OnClick(R.id.mainSettingsLayout)
    public void onBackgroundClick(View view) {
        SettingsManager.getInstance().closeSettings();
        finish();
        overridePendingTransition(0, 0);
    }

    @OnCheckedChanged({R.id.backCameraRadioBtn, R.id.frontCameraRadioBtn})
    public void onCameraSelected(CompoundButton button, boolean checked) {

        boolean backCamera = SettingsManager.getInstance().isBackCamera();
        if (checked) {
            switch (button.getId()) {
                case R.id.backCameraRadioBtn:
                    if (backCamera) return; //do not save or restart camera if no change
                    backCamera = true;
                    break;

                case R.id.frontCameraRadioBtn:
                    if (!backCamera) return;
                    backCamera = false;
                    break;
            }

            SettingsManager.getInstance().cameraChanged(backCamera);
        }
    }

    @OnCheckedChanged({R.id.pascalRadioBtn, R.id.faceRadioBtn})
    public void onModelSelected(CompoundButton button, boolean checked) {

        SSDModel ssdModel = SettingsManager.getInstance().getModel();
        if (checked) {
            switch (button.getId()) {
                case R.id.pascalRadioBtn:
                    if (ssdModel == SSDModel.PASCAL_MOBILENET_300) return;
                    ssdModel = SSDModel.PASCAL_MOBILENET_300;
                    break;

                case R.id.faceRadioBtn:
                    if (ssdModel == SSDModel.FACE_MOBILENET_160) return;
                    ssdModel = SSDModel.FACE_MOBILENET_160;
                    break;
            }

            SettingsManager.getInstance().modelChanged(ssdModel);
        }
    }

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.settings);
        ButterKnife.bind(this);
        populateSettingsView();
    }


    private void populateSettingsView() {
        switch (SettingsManager.getInstance().getModel()) {
            case PASCAL_MOBILENET_300:
                mPascalRadioBtn.setChecked(true);
                break;
            case FACE_MOBILENET_160:
                mFaceRadioBtn.setChecked(true);
                break;
        }

        if (SettingsManager.getInstance().isBackCamera()) {
            mBackCameraRadioBtn.setChecked(true);
        } else {
            mFrontCameraRadioBtn.setChecked(true);
        }

        float threshold = SettingsManager.getInstance().getThreshold();
        mAccuracySlider.setProgress((int)(threshold * 100));
        mAccuracyText.setText(threshold + "");

        mAccuracySlider.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                float accuracyThreshold = progress / 10 / 10.f;
                mAccuracyText.setText(accuracyThreshold + "");

                SettingsManager.getInstance().thresholdChanged(accuracyThreshold);
            }
            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
            }
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
            }
        });
    }
}
