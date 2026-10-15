package com.videogammetry;

import android.app.Activity;
import android.content.ContentValues;
import android.content.Intent;
import android.content.SharedPreferences;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.Manifest;
import android.content.pm.PackageManager;
import android.media.Image;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.provider.OpenableColumns;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.SeekBar;
import android.widget.Switch;
import android.widget.TextView;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.List;
import java.util.Locale;

// Debug app: tune the settings, pick a video, watch the point cloud being
// built from the camera's point of view, and get a .ply in Downloads/Videogammetry.
public class MainActivity extends Activity {

    private static final int PICK_VIDEO = 1;
    private static final int CAMERA_PERMISSION = 2;
    private static final int BATCH = 8;  // frames decoded per call

    private final Settings settings = new Settings();
    private SharedPreferences prefs;
    private LinearLayout settingsPanel;
    private Button pickButton, stopButton, cameraButton;
    private TextView statusText;
    private PointCloudView cloudView;
    private volatile boolean busy = false, stopRequested = false;

    private interface IntSetter { void set(int v); }
    private interface BoolSetter { void set(boolean v); }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        prefs = getSharedPreferences("settings", MODE_PRIVATE);
        settings.load(prefs);
        buildUi();
    }

    private int dp(int v) {
        return (int) (v * getResources().getDisplayMetrics().density);
    }

    private void buildUi() {
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(12), dp(24), dp(12), dp(12));

        TextView title = new TextView(this);
        title.setTextSize(20);
        title.setText("Videogammetry " + NativeBridge.version());
        root.addView(title);

        LinearLayout buttons = new LinearLayout(this);
        pickButton = new Button(this);
        pickButton.setText("Pick video");
        pickButton.setOnClickListener(v -> pickVideo());
        stopButton = new Button(this);
        stopButton.setText("Stop");
        stopButton.setEnabled(false);
        stopButton.setOnClickListener(v -> stopRequested = true);
        Button settingsButton = new Button(this);
        settingsButton.setText("Settings");
        settingsButton.setOnClickListener(v ->
                settingsPanel.setVisibility(settingsPanel.getVisibility() == View.VISIBLE ? View.GONE : View.VISIBLE));
        cameraButton = new Button(this);
        cameraButton.setText("Live camera");
        cameraButton.setOnClickListener(v -> startCamera());
        buttons.addView(cameraButton);
        buttons.addView(pickButton);
        buttons.addView(stopButton);
        buttons.addView(settingsButton);
        root.addView(buttons);

        statusText = new TextView(this);
        statusText.setTextSize(12);
        statusText.setText("Tune the settings, then start the live camera (or pick a video).\n"
                + "The camera image is not shown, only the 3D reconstruction from the camera's view.\n"
                + "The .ply is saved to Downloads/Videogammetry.");
        root.addView(statusText);

        cloudView = new PointCloudView(this);
        root.addView(cloudView, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));

        settingsPanel = new LinearLayout(this);
        settingsPanel.setOrientation(LinearLayout.VERTICAL);
        buildSettings(settingsPanel);
        ScrollView scroll = new ScrollView(this);
        scroll.addView(settingsPanel);
        root.addView(scroll, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1.2f));
        setContentView(root);
    }

    private void header(LinearLayout parent, String text) {
        TextView t = new TextView(this);
        t.setText(text);
        t.setTextSize(15);
        t.setPadding(0, dp(10), 0, dp(2));
        parent.addView(t);
    }

    // Slider over `values` (shown with `labels`), starting at the value closest to `current`.
    private void slider(LinearLayout parent, String name, String hint, int[] values, String[] labels, int current,
                        IntSetter setter) {
        TextView label = new TextView(this);
        label.setTextSize(13);
        SeekBar bar = new SeekBar(this);
        bar.setMax(values.length - 1);
        int start = 0;
        for (int i = 0; i < values.length; ++i) {
            if (Math.abs(values[i] - current) < Math.abs(values[start] - current)) start = i;
        }
        final int[] idx = {start};
        Runnable show = () -> label.setText(name + ": " + labels[idx[0]] + "   (" + hint + ")");
        bar.setProgress(start);
        show.run();
        bar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            public void onProgressChanged(SeekBar s, int p, boolean user) {
                idx[0] = p;
                setter.set(values[p]);
                show.run();
                settings.save(prefs);
            }
            public void onStartTrackingTouch(SeekBar s) {}
            public void onStopTrackingTouch(SeekBar s) {}
        });
        parent.addView(label);
        parent.addView(bar);
    }

    private void toggle(LinearLayout parent, String name, boolean current, BoolSetter setter) {
        Switch s = new Switch(this);
        s.setText(name);
        s.setTextSize(13);
        s.setChecked(current);
        s.setOnCheckedChangeListener((CompoundButton b, boolean on) -> {
            setter.set(on);
            settings.save(prefs);
        });
        parent.addView(s);
    }

    private static String[] labels(int[] values, String suffix, String zeroLabel) {
        String[] out = new String[values.length];
        for (int i = 0; i < values.length; ++i) out[i] = values[i] == 0 && zeroLabel != null ? zeroLabel : values[i] + suffix;
        return out;
    }

    private void buildSettings(LinearLayout p) {
        header(p, "Processing (speed vs detail)");
        int[] sizes = {480, 640, 800, 960, 1280, 0};
        slider(p, "Process size", "smaller = faster", sizes, labels(sizes, " px", "full"), settings.processSize,
                v -> settings.processSize = v);
        int[] feats = {500, 800, 1000, 1500, 2000, 3000};
        slider(p, "ORB features", "fewer = faster, less robust", feats, labels(feats, "", null), settings.maxFeatures,
                v -> settings.maxFeatures = v);
        int[] steps = {1, 2, 3, 4, 6};
        slider(p, "Use every N-th frame", "higher = faster, needs slow motion", steps, labels(steps, "", null),
                settings.frameStep, v -> settings.frameStep = v);
        int cores = Runtime.getRuntime().availableProcessors();
        int[] threads = {0, 1, 2, 4, 6, 8};
        slider(p, "Threads", cores + " cores here", threads, labels(threads, "", "all"), settings.threads,
                v -> settings.threads = v);
        int[] fovs = {45, 55, 65, 75, 85, 95};
        slider(p, "Starting field of view", "refined from the video if enabled", fovs, labels(fovs, " deg", null),
                Math.round(settings.fovDeg), v -> settings.fovDeg = v);

        header(p, "Mapping (accuracy vs cost)");
        int[] baKfs = {0, 5, 8, 10, 15};
        slider(p, "Local BA keyframes", "0 = off", baKfs, labels(baKfs, "", "off"), settings.localBAKeyframes,
                v -> settings.localBAKeyframes = v);
        int[] baIts = {3, 5, 10, 15};
        slider(p, "Local BA iterations", "fewer = faster", baIts, labels(baIts, "", null), settings.localBAIterations,
                v -> settings.localBAIterations = v);
        toggle(p, "Background global BA (fixes old parts of the map)", settings.backgroundBA,
                v -> settings.backgroundBA = v);
        toggle(p, "Estimate field of view from the video", settings.refineFocal, v -> settings.refineFocal = v);
        toggle(p, "Estimate lens distortion", settings.refineDistortion, v -> settings.refineDistortion = v);
        toggle(p, "Fuse duplicate points (less drift, more cost)", settings.fusion, v -> settings.fusion = v);
        toggle(p, "Keep only points confirmed by 3 views (less noise)", settings.confirmPoints,
                v -> settings.confirmPoints = v);

        header(p, "Live view");
        toggle(p, "Show the point cloud while processing", settings.liveView, v -> settings.liveView = v);
        int[] vfps = {5, 10, 15, 20, 30};
        slider(p, "View updates", "per second", vfps, labels(vfps, " /s", null), settings.viewFps,
                v -> settings.viewFps = v);
        int[] ps = {1, 2, 3, 4, 6};
        slider(p, "Point size", "pixels", ps, labels(ps, " px", null), settings.pointSize, v -> settings.pointSize = v);
        toggle(p, "Tracked points in green", settings.highlightTracked, v -> settings.highlightTracked = v);

        Button reset = new Button(this);
        reset.setText("Reset to defaults");
        reset.setOnClickListener(v -> {
            settings.reset();
            settings.save(prefs);
            buildUi();
        });
        p.addView(reset);
    }

    private void startCamera() {
        if (busy) return;
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[] {Manifest.permission.CAMERA}, CAMERA_PERMISSION);
            return;
        }
        beginRun();
        new Thread(this::processCamera, "reconstruction").start();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] results) {
        if (requestCode == CAMERA_PERMISSION && results.length > 0 && results[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else if (requestCode == CAMERA_PERMISSION) {
            show("Camera permission is needed for the live mode.");
        }
    }

    private void beginRun() {
        busy = true;
        stopRequested = false;
        pickButton.setEnabled(false);
        cameraButton.setEnabled(false);
        stopButton.setEnabled(true);
        settingsPanel.setVisibility(View.GONE);
        cloudView.clear();
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }

    private void endRun() {
        runOnUiThread(() -> {
            busy = false;
            pickButton.setEnabled(true);
            cameraButton.setEnabled(true);
            stopButton.setEnabled(false);
            getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        });
    }

    // Live camera: always takes the newest frame when the reconstruction is
    // free; frames that arrive meanwhile are dropped (no queue, no lag).
    private void processCamera() {
        long session = 0;
        CameraStream camera = new CameraStream(this);
        try {
            int processSize = settings.processSize > 0 ? settings.processSize : 1920;
            if (!camera.choose(processSize)) throw new IllegalStateException("no back camera");
            float fov = camera.fovDeg > 20 && camera.fovDeg < 150 ? camera.fovDeg : settings.fovDeg;
            float[] floats = settings.toFloats();
            floats[0] = fov;
            session = NativeBridge.create(settings.toInts(), floats);
            cloudView.setImageRotation(camera.sensorOrientation);
            camera.start();

            long start = SystemClock.elapsedRealtime(), lastView = 0, lastStatus = 0, processMs = 0;
            int processed = 0;
            while (!stopRequested) {
                if (camera.error() != null) throw new IllegalStateException(camera.error());
                Image image = camera.latest();
                if (image == null) {
                    SystemClock.sleep(2);
                    continue;
                }
                long t0 = SystemClock.elapsedRealtime();
                Image.Plane[] planes = image.getPlanes();
                NativeBridge.addFrameYuv(session, planes[0].getBuffer(), planes[1].getBuffer(), planes[2].getBuffer(),
                        image.getWidth(), image.getHeight(), planes[0].getRowStride(), planes[1].getRowStride(),
                        planes[1].getPixelStride(), processSize, image.getTimestamp() * 1e-9);
                image.close();
                processMs += SystemClock.elapsedRealtime() - t0;
                ++processed;

                long now = SystemClock.elapsedRealtime();
                if (settings.liveView && now - lastView >= 1000 / Math.max(1, settings.viewFps)) {
                    Bitmap target = cloudView.backBuffer(true);
                    if (target != null) {
                        NativeBridge.renderView(session, target, settings.pointSize, settings.highlightTracked);
                        cloudView.swap();
                    }
                    lastView = now;
                }
                if (now - lastStatus >= 250) {
                    double seconds = Math.max((now - start) / 1000.0, 1e-3);
                    long delivered = camera.delivered();
                    show(String.format(Locale.US,
                            "Live camera %dx%d, starting FOV %.1f deg%s\n"
                                    + "camera %.1f fps, processed %.1f fps (%.1f ms each), skipped %d\n%s",
                            camera.size.getWidth(), camera.size.getHeight(), fov,
                            camera.fovDeg > 0 ? " (from camera)" : " (setting)",
                            delivered / seconds, processed / seconds, (double) processMs / Math.max(1, processed),
                            Math.max(0, delivered - processed), NativeBridge.status(session)));
                    lastStatus = now;
                }
            }
            camera.stop();
            saveResult(session, "camera_" + new java.text.SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US)
                    .format(new java.util.Date()), start);
        } catch (Exception e) {
            show("Error: " + e.getMessage());
        } finally {
            camera.stop();
            if (session != 0) NativeBridge.destroy(session);
            endRun();
        }
    }

    // Final global BA, last view, .ply into Downloads/Videogammetry.
    private void saveResult(long session, String baseName, long start) throws Exception {
        show("Refining the whole map (bundle adjustment)...\n" + NativeBridge.status(session));
        NativeBridge.finish(session);
        Bitmap target = cloudView.backBuffer(true);
        if (target != null) {
            NativeBridge.renderView(session, target, settings.pointSize, false);
            cloudView.swap();
        }
        File tmp = new File(getCacheDir(), "cloud.ply");
        int points = NativeBridge.writePly(session, tmp.getAbsolutePath());
        if (points < 0) throw new IllegalStateException("could not write the point cloud");
        String outName = baseName + "_cloud.ply";
        saveToDownloads(tmp, outName);
        tmp.delete();
        double seconds = (SystemClock.elapsedRealtime() - start) / 1000.0;
        show(String.format(Locale.US, "Done after %.1f s: %d points\nSaved to Downloads/Videogammetry/%s\n%s",
                seconds, points, outName, NativeBridge.status(session)));
    }

    private void pickVideo() {
        if (busy) return;
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("video/*");
        startActivityForResult(intent, PICK_VIDEO);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != PICK_VIDEO || resultCode != RESULT_OK || data == null || data.getData() == null) return;
        Uri uri = data.getData();
        beginRun();
        cloudView.setImageRotation(0);
        new Thread(() -> process(uri), "reconstruction").start();
    }

    private void process(Uri uri) {
        long session = 0;
        MediaMetadataRetriever retriever = new MediaMetadataRetriever();
        try {
            retriever.setDataSource(this, uri);
            int frameCount = parseInt(retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT));
            long durationMs = parseInt(retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION));
            if (frameCount <= 0) throw new IllegalStateException("could not read the video's frame count");
            double fps = durationMs > 0 ? frameCount * 1000.0 / durationMs : 30.0;
            String name = displayName(uri);

            session = NativeBridge.create(settings.toInts(), settings.toFloats());
            MediaMetadataRetriever.BitmapParams params = new MediaMetadataRetriever.BitmapParams();
            params.setPreferredConfig(Bitmap.Config.ARGB_8888);

            long start = SystemClock.elapsedRealtime(), lastView = 0, lastStatus = 0;
            long processMs = 0;
            int processed = 0;
            for (int first = 0; first < frameCount && !stopRequested; first += BATCH) {
                int count = Math.min(BATCH, frameCount - first);
                List<Bitmap> frames = retriever.getFramesAtIndex(first, count, params);
                for (int k = 0; k < frames.size(); ++k) {
                    Bitmap frame = frames.get(k);
                    int index = first + k;
                    if (index % settings.frameStep == 0 && !stopRequested) {
                        if (frame.getConfig() != Bitmap.Config.ARGB_8888) {
                            Bitmap converted = frame.copy(Bitmap.Config.ARGB_8888, false);
                            frame.recycle();
                            frame = converted;
                        }
                        long t0 = SystemClock.elapsedRealtime();
                        NativeBridge.addFrame(session, frame, index / fps);
                        processMs += SystemClock.elapsedRealtime() - t0;
                        ++processed;
                    }
                    frame.recycle();
                }
                long now = SystemClock.elapsedRealtime();
                if (settings.liveView && now - lastView >= 1000 / Math.max(1, settings.viewFps)) {
                    Bitmap target = cloudView.backBuffer(true);
                    if (target != null) {
                        NativeBridge.renderView(session, target, settings.pointSize, settings.highlightTracked);
                        cloudView.swap();
                    }
                    lastView = now;
                }
                if (now - lastStatus >= 250) {
                    double seconds = (now - start) / 1000.0;
                    int done = first + count;
                    show(String.format(Locale.US, "%s  frame %d / %d\n%.1f fps overall, %.1f ms per processed frame\n%s",
                            name, done, frameCount, done / Math.max(seconds, 1e-3),
                            processed > 0 ? (double) processMs / processed : 0.0, NativeBridge.status(session)));
                    lastStatus = now;
                }
            }

            saveResult(session, stripExtension(name), start);
        } catch (Exception e) {
            show("Error: " + e.getMessage());
        } finally {
            try {
                retriever.release();
            } catch (Exception ignored) {
            }
            if (session != 0) NativeBridge.destroy(session);
            endRun();
        }
    }

    private void saveToDownloads(File file, String name) throws Exception {
        ContentValues values = new ContentValues();
        values.put(MediaStore.Downloads.DISPLAY_NAME, name);
        values.put(MediaStore.Downloads.MIME_TYPE, "application/octet-stream");
        values.put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/Videogammetry");
        Uri target = getContentResolver().insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values);
        if (target == null) throw new IllegalStateException("could not create the file in Downloads");
        try (InputStream in = new FileInputStream(file);
             OutputStream out = getContentResolver().openOutputStream(target)) {
            byte[] buffer = new byte[1 << 16];
            int n;
            while ((n = in.read(buffer)) > 0) out.write(buffer, 0, n);
        }
    }

    private String displayName(Uri uri) {
        try (Cursor c = getContentResolver().query(uri, new String[]{OpenableColumns.DISPLAY_NAME}, null, null, null)) {
            if (c != null && c.moveToFirst()) return c.getString(0);
        } catch (Exception ignored) {
        }
        return "video";
    }

    private static String stripExtension(String name) {
        int dot = name.lastIndexOf('.');
        return dot > 0 ? name.substring(0, dot) : name;
    }

    private static int parseInt(String s) {
        try {
            return s == null ? 0 : Integer.parseInt(s.trim());
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    private void show(String text) {
        runOnUiThread(() -> statusText.setText(text));
    }
}
