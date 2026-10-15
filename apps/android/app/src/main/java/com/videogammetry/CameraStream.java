package com.videogammetry;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Size;
import android.util.SizeF;

import java.util.Collections;

// Back camera streaming YUV frames into an ImageReader, with no preview.
// The consumer takes only the newest frame (latest()); older ones are
// dropped by the reader, so processing never queues behind the camera.
final class CameraStream {
    private final CameraManager manager;
    private HandlerThread thread;
    private Handler handler;
    private CameraDevice device;
    private CameraCaptureSession session;
    private ImageReader reader;
    private volatile String error;
    private volatile long delivered = 0;  // frames the camera produced

    String cameraId;
    Size size;
    int sensorOrientation;  // degrees the image must be rotated to look upright (portrait)
    float fovDeg = 0;       // across the long side, from the camera's characteristics (0 = unknown)

    CameraStream(Context context) {
        manager = (CameraManager) context.getSystemService(Context.CAMERA_SERVICE);
    }

    // Picks the back camera and the smallest YUV size whose long side is at
    // least `minLongSide` (or the largest available). False if no camera.
    boolean choose(int minLongSide) throws CameraAccessException {
        for (String id : manager.getCameraIdList()) {
            CameraCharacteristics c = manager.getCameraCharacteristics(id);
            Integer facing = c.get(CameraCharacteristics.LENS_FACING);
            if (facing == null || facing != CameraCharacteristics.LENS_FACING_BACK) continue;
            StreamConfigurationMap map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            if (map == null) continue;
            Size best = null, largest = null;
            for (Size s : map.getOutputSizes(ImageFormat.YUV_420_888)) {
                int longSide = Math.max(s.getWidth(), s.getHeight());
                if (largest == null || longSide > Math.max(largest.getWidth(), largest.getHeight())) largest = s;
                if (longSide >= minLongSide &&
                        (best == null || longSide < Math.max(best.getWidth(), best.getHeight()))) best = s;
            }
            cameraId = id;
            size = best != null ? best : largest;
            Integer orientation = c.get(CameraCharacteristics.SENSOR_ORIENTATION);
            sensorOrientation = orientation != null ? orientation : 90;
            fovDeg = estimateFov(c);
            return size != null;
        }
        return false;
    }

    // Field of view across the long side of the output: from the intrinsic
    // calibration if the camera reports it, else from focal length and sensor
    // size (the output uses the full sensor width).
    private float estimateFov(CameraCharacteristics c) {
        Rect active = c.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE);
        float[] intrinsics = c.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION);
        if (active != null && intrinsics != null && intrinsics.length >= 1 && intrinsics[0] > 0) {
            return (float) Math.toDegrees(2 * Math.atan(0.5 * active.width() / intrinsics[0]));
        }
        float[] focal = c.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS);
        SizeF sensor = c.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE);
        if (focal != null && focal.length > 0 && sensor != null && focal[0] > 0) {
            return (float) Math.toDegrees(2 * Math.atan(0.5 * Math.max(sensor.getWidth(), sensor.getHeight()) / focal[0]));
        }
        return 0;
    }

    @SuppressLint("MissingPermission")
    void start() throws CameraAccessException {
        thread = new HandlerThread("camera");
        thread.start();
        handler = new Handler(thread.getLooper());
        reader = ImageReader.newInstance(size.getWidth(), size.getHeight(), ImageFormat.YUV_420_888, 3);
        reader.setOnImageAvailableListener(r -> ++delivered, handler);
        manager.openCamera(cameraId, new CameraDevice.StateCallback() {
            @Override
            public void onOpened(CameraDevice camera) {
                device = camera;
                try {
                    camera.createCaptureSession(Collections.singletonList(reader.getSurface()),
                            new CameraCaptureSession.StateCallback() {
                                @Override
                                public void onConfigured(CameraCaptureSession s) {
                                    session = s;
                                    try {
                                        CaptureRequest.Builder b = camera.createCaptureRequest(CameraDevice.TEMPLATE_RECORD);
                                        b.addTarget(reader.getSurface());
                                        b.set(CaptureRequest.CONTROL_AF_MODE,
                                                CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO);
                                        s.setRepeatingRequest(b.build(), null, handler);
                                    } catch (Exception e) {
                                        error = "capture: " + e.getMessage();
                                    }
                                }

                                @Override
                                public void onConfigureFailed(CameraCaptureSession s) {
                                    error = "could not configure the camera";
                                }
                            }, handler);
                } catch (Exception e) {
                    error = "session: " + e.getMessage();
                }
            }

            @Override
            public void onDisconnected(CameraDevice camera) {
                error = "camera disconnected";
                camera.close();
            }

            @Override
            public void onError(CameraDevice camera, int code) {
                error = "camera error " + code;
                camera.close();
            }
        }, handler);
    }

    // Newest frame (close it when done), or null if none since the last call.
    Image latest() {
        try {
            return reader != null ? reader.acquireLatestImage() : null;
        } catch (IllegalStateException e) {
            return null;
        }
    }

    long delivered() {
        return delivered;
    }

    String error() {
        return error;
    }

    void stop() {
        try {
            if (session != null) session.close();
        } catch (Exception ignored) {
        }
        if (device != null) device.close();
        if (reader != null) reader.close();
        if (thread != null) thread.quitSafely();
        session = null;
        device = null;
        reader = null;
    }
}
