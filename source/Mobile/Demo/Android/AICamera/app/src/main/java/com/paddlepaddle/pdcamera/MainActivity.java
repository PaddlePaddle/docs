package com.paddlepaddle.pdcamera;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureFailure;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Process;
import android.support.annotation.NonNull;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Toast;

import java.util.Arrays;
import java.util.List;

import butterknife.BindView;
import butterknife.ButterKnife;
import butterknife.OnClick;

public class MainActivity extends AppCompatActivity implements SettingsManager.SettingsListener {

    private static final String TAG = MainActivity.class.getSimpleName();

    private static final int PERMISSIONS_REQUEST = 1;

    private CameraCaptureSession mCaptureSession;
    private CameraDevice mCameraDevice;

    private HandlerThread mCaptureThread; //background thread for capturing image
    private Handler mCaptureHandler;
    private HandlerThread mInferThread; //background thread for inferencing from paddle
    private Handler mInferHandler;

    private ImageReader mImageReader;
    private ImageRecognizer mImageRecognizer;

    private Size mPreviewSize;
    private byte[] mRgbBytes;

    private boolean mInProcessing;
    private boolean mCapturing;

    @BindView(R.id.textureView)
    AutoFitTextureView mTextureView;

    @BindView(R.id.ssdLayerView)
    SSDLayerView mSSDLayerView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        ButterKnife.bind(this);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        SettingsManager.getInstance().loadSettings(this);
        SettingsManager.getInstance().setListener(this);

        mImageRecognizer = new ImageRecognizer(this, SettingsManager.getInstance().getModel());

    }

    private boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED &&
                    checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED;
        } else {
            return true;
        }
    }

    private void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (shouldShowRequestPermissionRationale(Manifest.permission.CAMERA) ||
                    shouldShowRequestPermissionRationale(Manifest.permission.WRITE_EXTERNAL_STORAGE)) {
                Toast.makeText(MainActivity.this,
                        "Camera AND storage permission are required for this demo", Toast.LENGTH_LONG).show();
            }
            requestPermissions(new String[]{Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE}, PERMISSIONS_REQUEST);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == PERMISSIONS_REQUEST) {
            if (grantResults.length > 0
                    && grantResults[0] == PackageManager.PERMISSION_GRANTED
                    && grantResults[1] == PackageManager.PERMISSION_GRANTED) {
                startCapture();
            } else {
                requestPermission();
            }
        }
    }

    @Override
    protected void onStart() {
        super.onStart();

        startCaptureThread();
        startInferThread();

        if (mTextureView.isAvailable()) {
            startCapture();
        } else {
            mTextureView.setSurfaceTextureListener(new TextureView.SurfaceTextureListener() {
                @Override
                public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height) {
                    startCapture();
                }

                @Override
                public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height) { }

                @Override
                public boolean onSurfaceTextureDestroyed(SurfaceTexture surface) {
                    return true;
                }

                @Override
                public void onSurfaceTextureUpdated(SurfaceTexture surface) { }
            });
        }
    }

    @Override
    protected void onStop() {
        closeCamera();
        stopCaptureThread();
        stopInferThread();
        super.onStop();
    }

    private void startCaptureThread() {
        mCaptureThread = new HandlerThread("capture");
        mCaptureThread.start();
        mCaptureHandler = new Handler(mCaptureThread.getLooper());
    }

    private void startInferThread() {
        mInferThread = new HandlerThread("inference");
        mInferThread.start();
        mInferHandler = new Handler(mInferThread.getLooper());
    }

    private void stopCaptureThread() {
        mCaptureThread.quitSafely();
        try {
            mCaptureThread.join();
            mCaptureThread = null;
            mCaptureHandler = null;
        } catch (final InterruptedException e) {
        }
    }

    private void stopInferThread() {
        mInferThread.quitSafely();
        try {
            mInferThread.join();
            mInferThread = null;
            mInferHandler = null;
        } catch (final InterruptedException e) {
        }
    }

    private void startCapture() {
        if (!hasPermission()) {
            requestPermission();
            return;
        }

        if (mCapturing) return;

        mCapturing = true;

        final CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);

        String cameraIdAvailable = null;
        try {
            for (final String cameraId : manager.getCameraIdList()) {
                final CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);

                final Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
                if (facing != null && facing == (SettingsManager.getInstance().isBackCamera() ? CameraCharacteristics.LENS_FACING_BACK : CameraCharacteristics.LENS_FACING_FRONT)) {
                    cameraIdAvailable = cameraId;
                    break;
                }
            }
        } catch (CameraAccessException e) {
            android.util.Log.e(TAG, "Start Capture exception", e);
        }

        try {
            final CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraIdAvailable);

            final StreamConfigurationMap map =
                    characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            mPreviewSize = ImageUtils.chooseOptimalSize(map.getOutputSizes(SurfaceTexture.class),
                    mTextureView.getHeight() / 2,
                    mTextureView.getWidth() / 2);
            mTextureView.setAspectRatio(
                    mPreviewSize.getHeight(), mPreviewSize.getWidth());
            float aspectRatio = mPreviewSize.getWidth() * 1.0f / mPreviewSize.getHeight();
            mSSDLayerView.setTextureViewDimen(mTextureView.getWidth(), (int) (mTextureView.getWidth() * aspectRatio));

            manager.openCamera(cameraIdAvailable, new CameraDevice.StateCallback() {
                @Override
                public void onOpened(@NonNull CameraDevice camera) {
                    mCameraDevice = camera;
                    createCaptureSession();
                }

                @Override
                public void onDisconnected(@NonNull CameraDevice camera) {
                    camera.close();
                    mCameraDevice = null;
                    mCapturing = false;
                }

                @Override
                public void onError(@NonNull CameraDevice camera, final int error) {
                    android.util.Log.e(TAG, "open Camera on Error =  " + error);
                    camera.close();
                    mCameraDevice = null;
                    mCapturing = false;
                }
            }, mCaptureHandler);
        } catch (CameraAccessException e) {
            mCapturing = false;
            android.util.Log.e(TAG, "Start Capture exception", e);
        } catch (SecurityException e) {
            mCapturing = false;
            android.util.Log.e(TAG, "Start Capture exception", e);
        }
    }

    private void createCaptureSession() {
        try {
            final SurfaceTexture texture = mTextureView.getSurfaceTexture();
            texture.setDefaultBufferSize(mPreviewSize.getWidth(), mPreviewSize.getHeight());

            final Surface surface = new Surface(texture);
            final CaptureRequest.Builder captureRequestBuilder = mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            captureRequestBuilder.addTarget(surface);

            mImageReader = ImageReader.newInstance(
                    mPreviewSize.getWidth(), mPreviewSize.getHeight(), ImageFormat.YUV_420_888, 10);

            mImageReader.setOnImageAvailableListener(new ImageReader.OnImageAvailableListener() {
                @Override
                public void onImageAvailable(ImageReader reader) {
                    final int previewWidth = mPreviewSize.getWidth();
                    final int previewHeight = mPreviewSize.getHeight();
                    if (previewWidth == 0 || previewHeight == 0) {
                        return;
                    }
                    if (mRgbBytes == null) {
                        mRgbBytes = new byte[previewWidth * previewHeight * 3];
                    }

                    final Image image = reader.acquireNextImage();
                    if (image == null) return;

                    if (mInProcessing || mInferHandler == null) {
                        image.close();
                        return;
                    }

                    mInProcessing = true;

                    final byte[][] yuvBytes = new byte[3][];
                    final Image.Plane[] planes = image.getPlanes();
                    ImageUtils.fillBytes(planes, yuvBytes);
                    final int yRowStride = planes[0].getRowStride();
                    final int uvRowStride = planes[1].getRowStride();
                    final int uvPixelStride = planes[1].getPixelStride();

                    mInferHandler.post(new Runnable() {
                        @Override
                        public void run() {
                            if (mImageRecognizer != null) {
                                ImageUtils.convertYUV420ToARGB8888(
                                        yuvBytes[0],
                                        yuvBytes[1],
                                        yuvBytes[2],
                                        previewWidth,
                                        previewHeight,
                                        yRowStride,
                                        uvRowStride,
                                        uvPixelStride,
                                        mRgbBytes);

                                List<SSDData> results = mImageRecognizer.infer(mRgbBytes, previewHeight, previewWidth, 3, SettingsManager.getInstance().getThreshold(), SettingsManager.getInstance().isBackCamera());

                                mSSDLayerView.populateSSDList(results, SettingsManager.getInstance().getModel() != SSDModel.FACE_MOBILENET_160);
                            }
                            image.close();
                            mInProcessing = false;
                        }
                    });
                }
            }, mCaptureHandler);
            captureRequestBuilder.addTarget(mImageReader.getSurface());

            mCameraDevice.createCaptureSession(
                    Arrays.asList(surface, mImageReader.getSurface()),
                    new CameraCaptureSession.StateCallback() {

                        @Override
                        public void onConfigured(final CameraCaptureSession cameraCaptureSession) {
                            if (null == mCameraDevice) {
                                return;
                            }

                            mCaptureSession = cameraCaptureSession;
                            try {
                                captureRequestBuilder.set(
                                        CaptureRequest.CONTROL_AF_MODE,
                                        CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);
                                captureRequestBuilder.set(
                                        CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON_AUTO_FLASH);

                                CaptureRequest previewRequest = captureRequestBuilder.build();
                                mCaptureSession.setRepeatingRequest(
                                        previewRequest, new CameraCaptureSession.CaptureCallback() {
                                            @Override
                                            public void onCaptureProgressed(@NonNull CameraCaptureSession session, @NonNull CaptureRequest request, @NonNull CaptureResult partialResult) {
                                                super.onCaptureProgressed(session, request, partialResult);
                                            }

                                            @Override
                                            public void onCaptureFailed(@NonNull CameraCaptureSession session, @NonNull CaptureRequest request, @NonNull CaptureFailure failure) {
                                                super.onCaptureFailed(session, request, failure);
                                                Log.d(TAG, "onCaptureFailed = " + failure.getReason());
                                            }

                                            @Override
                                            public void onCaptureSequenceCompleted(@NonNull CameraCaptureSession session, int sequenceId, long frameNumber) {
                                                super.onCaptureSequenceCompleted(session, sequenceId, frameNumber);
                                                Log.d(TAG, "onCaptureSequenceCompleted");
                                            }
                                        }, mCaptureHandler);
                            } catch (final CameraAccessException e) {
                                Log.e(TAG, "onConfigured exception ", e);
                            }
                        }

                        @Override
                        public void onConfigureFailed(final CameraCaptureSession cameraCaptureSession) {
                            Log.e(TAG, "onConfigureFailed ");
                        }
                    },
                    null);
        } catch (final CameraAccessException e) {
            Log.e(TAG, "createCaptureSession exception ", e);
        }
    }

    private void closeCamera() {
        if (mCaptureSession != null) {
            mCaptureSession.close();
            mCaptureSession = null;
        }
        if (mCameraDevice != null) {
            mCameraDevice.close();
            mCameraDevice = null;
        }
        if (mImageReader != null) {
            mImageReader.close();
            mImageReader = null;
        }

        mCapturing = false;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        Process.killProcess(Process.myPid());
    }

    @OnClick(R.id.ssdLayerView)
    public void onLayerClick(View v) {
        startActivity(new Intent(this, SettingsActivity.class));
        overridePendingTransition(0, 0);
    }

    @Override
    public void onCameraChanged(boolean backCamera) {
        closeCamera();
        startCapture();
    }

    @Override
    public void onModelChanged(SSDModel ssdModel) {
        closeCamera();
        stopCaptureThread();
        stopInferThread();
        Process.killProcess(Process.myPid());
    }

    @Override
    public void onThresholdChanged(float threshold) {
    }

    @Override
    public void onSettingsClose() {
    }
}
