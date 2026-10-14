package com.videogammetry;

import android.app.Activity;
import android.content.ContentValues;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.provider.OpenableColumns;
import android.view.Gravity;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.List;
import java.util.Locale;

// Debug app: pick a video, reconstruct a point cloud on the phone, and save
// it as a .ply file in Downloads/Videogammetry.
public class MainActivity extends Activity {

    private static final int PICK_VIDEO = 1;
    private static final int PROCESS_SIZE = 960;  // long side, as on desktop
    private static final double FOV_DEG = 65.0;   // assumed, across the long side
    private static final int BATCH = 8;           // frames decoded per call

    private Button pickButton;
    private TextView statusText;
    private volatile boolean busy = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setGravity(Gravity.CENTER_HORIZONTAL);
        int pad = (int) (24 * getResources().getDisplayMetrics().density);
        layout.setPadding(pad, pad * 2, pad, pad);

        TextView title = new TextView(this);
        title.setTextSize(22);
        title.setText("Videogammetry " + NativeBridge.version());
        layout.addView(title);

        pickButton = new Button(this);
        pickButton.setText("Pick video");
        pickButton.setOnClickListener(v -> pickVideo());
        layout.addView(pickButton);

        statusText = new TextView(this);
        statusText.setTextSize(15);
        statusText.setPadding(0, pad, 0, 0);
        statusText.setText("Pick a video to build a point cloud.\nThe .ply is saved to Downloads/Videogammetry.");
        layout.addView(statusText);

        setContentView(layout);
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
        busy = true;
        pickButton.setEnabled(false);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
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

            session = NativeBridge.create(PROCESS_SIZE, FOV_DEG);
            MediaMetadataRetriever.BitmapParams params = new MediaMetadataRetriever.BitmapParams();
            params.setPreferredConfig(Bitmap.Config.ARGB_8888);

            long start = SystemClock.elapsedRealtime();
            for (int first = 0; first < frameCount; first += BATCH) {
                int count = Math.min(BATCH, frameCount - first);
                List<Bitmap> frames = retriever.getFramesAtIndex(first, count, params);
                for (int k = 0; k < frames.size(); ++k) {
                    Bitmap frame = frames.get(k);
                    if (frame.getConfig() != Bitmap.Config.ARGB_8888) {
                        Bitmap converted = frame.copy(Bitmap.Config.ARGB_8888, false);
                        frame.recycle();
                        frame = converted;
                    }
                    NativeBridge.addFrame(session, frame, (first + k) / fps);
                    frame.recycle();
                }
                double seconds = (SystemClock.elapsedRealtime() - start) / 1000.0;
                int done = first + count;
                show(String.format(Locale.US, "%s\nframe %d / %d  (%.1f fps)\n%s", name, done, frameCount,
                        done / Math.max(seconds, 1e-3), NativeBridge.status(session)));
            }

            show("Refining the whole map (bundle adjustment)...\n" + NativeBridge.status(session));
            NativeBridge.finish(session);

            File tmp = new File(getCacheDir(), "cloud.ply");
            int points = NativeBridge.writePly(session, tmp.getAbsolutePath());
            if (points < 0) throw new IllegalStateException("could not write the point cloud");
            String outName = stripExtension(name) + "_cloud.ply";
            saveToDownloads(tmp, outName);
            tmp.delete();

            double seconds = (SystemClock.elapsedRealtime() - start) / 1000.0;
            show(String.format(Locale.US, "Done in %.1f s: %d points\nSaved to Downloads/Videogammetry/%s\n%s",
                    seconds, points, outName, NativeBridge.status(session)));
        } catch (Exception e) {
            show("Error: " + e.getMessage());
        } finally {
            try {
                retriever.release();
            } catch (Exception ignored) {
            }
            if (session != 0) NativeBridge.destroy(session);
            runOnUiThread(() -> {
                busy = false;
                pickButton.setEnabled(true);
                getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
            });
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
