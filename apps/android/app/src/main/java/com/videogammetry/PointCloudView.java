package com.videogammetry;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.view.View;

// Shows the point cloud rendered (natively, on the processing thread) from
// the current camera. Double-buffered: the processing thread draws into the
// back bitmap and swaps; onDraw shows the front one.
final class PointCloudView extends View {
    private final Object lock = new Object();
    private Bitmap front, back;
    private final Paint paint = new Paint(Paint.FILTER_BITMAP_FLAG);
    private int width, height;
    private volatile int rotation = 0;  // degrees the rendered image is rotated for display

    void setImageRotation(int degrees) {
        rotation = ((degrees % 360) + 360) % 360;
    }

    PointCloudView(Context context) {
        super(context);
        setBackgroundColor(Color.rgb(22, 24, 29));
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        synchronized (lock) {
            width = w;
            height = h;
        }
    }

    // For the processing thread: a bitmap of the view's size (half
    // resolution when `half`) to render into, or null if not laid out yet.
    Bitmap backBuffer(boolean half) {
        synchronized (lock) {
            int vw = rotation % 180 == 0 ? width : height, vh = rotation % 180 == 0 ? height : width;
            int w = half ? vw / 2 : vw, h = half ? vh / 2 : vh;
            if (w <= 0 || h <= 0) return null;
            if (back == null || back.getWidth() != w || back.getHeight() != h) {
                back = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
            }
            return back;
        }
    }

    // For the processing thread: shows the back buffer.
    void swap() {
        synchronized (lock) {
            Bitmap t = front;
            front = back;
            back = t;
        }
        postInvalidate();
    }

    void clear() {
        synchronized (lock) {
            front = null;
        }
        postInvalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        synchronized (lock) {
            if (front == null) return;
            canvas.save();
            canvas.translate(getWidth() / 2f, getHeight() / 2f);
            canvas.rotate(rotation);
            boolean swap = rotation % 180 != 0;
            int w = swap ? getHeight() : getWidth(), h = swap ? getWidth() : getHeight();
            canvas.drawBitmap(front, null, new android.graphics.Rect(-w / 2, -h / 2, w / 2, h / 2), paint);
            canvas.restore();
        }
    }
}
