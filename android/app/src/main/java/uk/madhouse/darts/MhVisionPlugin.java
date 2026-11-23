package uk.madhouse.darts;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.providers.NNAPIFlags;
import android.content.ContentValues;
import android.content.SharedPreferences;
import android.net.Uri;
import android.os.Build;
import android.os.Environment;
import android.provider.MediaStore;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Base64;
import android.util.Log;
import com.getcapacitor.JSArray;
import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.EnumSet;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Runs the Dart Sense model natively, on the phone's AI chip where it helps.
 *
 * The model is not built into the app (to keep the download small): the first time the camera
 * is used it is downloaded from the Madhouse site and kept on the phone. The fastest engine is
 * remembered, so the timing test only runs once.
 *
 * At start-up it tries each engine this phone offers (the AI chip through Android's neural
 * networks API, then the phone's own processor two ways), times a few test runs of each, and
 * keeps whichever is fastest. Madhouse's web code sends it the same 800x800 picture it would
 * otherwise score itself and gets back the raw candidates, then carries on exactly as before.
 * If any of this fails, Madhouse falls back to its normal in-browser engine.
 */
@CapacitorPlugin(name = "MhVision")
public class MhVisionPlugin extends Plugin {

    private static final String TAG = "MhVision";
    private static final int S = 800;
    private static final float MIN_CONF = 0.25f;

    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private OrtEnvironment env;
    private OrtSession session;
    private String inputName;
    private String engine = "";
    private String report = "";
    private double engineMs = 0;
    private FloatBuffer input;
    private volatile long dlGot = 0, dlTotal = 0;
    private static final String MODEL_FILE = "darts-sense-v1.onnx";
    private final int[] pixels = new int[S * S];

    @PluginMethod
    public void load(final PluginCall call) {
        final String url = call.getString("url");
        worker.execute(() -> {
            try {
                if (session == null) init(url);
                JSObject r = new JSObject();
                r.put("engine", engine);
                r.put("ms", Math.round(engineMs));
                r.put("report", report);
                call.resolve(r);
            } catch (Throwable t) {
                Log.e(TAG, "load failed", t);
                call.reject("Native engine unavailable: " + t.getMessage());
            }
        });
    }

    // Save a text file (backup, camera report) into the phone's Downloads folder.
    @PluginMethod
    public void saveFile(final PluginCall call) {
        final String name = call.getString("name", "madhouse.json");
        final String text = call.getString("text", "");
        final String mime = call.getString("mime", "application/json");
        new Thread(() -> {
            try {
                byte[] data = text.getBytes(StandardCharsets.UTF_8);
                String where;
                if (Build.VERSION.SDK_INT >= 29) {
                    ContentValues v = new ContentValues();
                    v.put(MediaStore.MediaColumns.DISPLAY_NAME, name);
                    v.put(MediaStore.MediaColumns.MIME_TYPE, mime);
                    v.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS);
                    Uri u = getContext().getContentResolver().insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, v);
                    if (u == null) throw new IllegalStateException("Couldn't create the file");
                    try (OutputStream o = getContext().getContentResolver().openOutputStream(u)) {
                        if (o == null) throw new IllegalStateException("Couldn't open the file");
                        o.write(data);
                    }
                    where = "Download/" + name;
                } else {
                    File d = getContext().getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS);
                    if (d == null) d = getContext().getFilesDir();
                    if (!d.exists()) d.mkdirs();
                    File f = new File(d, name);
                    try (FileOutputStream o = new FileOutputStream(f)) {
                        o.write(data);
                    }
                    where = f.getAbsolutePath();
                }
                JSObject r = new JSObject();
                r.put("where", where);
                call.resolve(r);
            } catch (Throwable t) {
                Log.e(TAG, "saveFile failed", t);
                call.reject("Couldn't save: " + t.getMessage());
            }
        }).start();
    }

    // How far the first-time model download has got (answered straight away, not queued behind the download)
    @PluginMethod
    public void progress(final PluginCall call) {
        JSObject r = new JSObject();
        r.put("got", dlGot);
        r.put("total", dlTotal);
        call.resolve(r);
    }

    @PluginMethod
    public void detect(final PluginCall call) {
        final String img = call.getString("img");
        if (img == null) {
            call.reject("No picture");
            return;
        }
        worker.execute(() -> {
            try {
                if (session == null) throw new IllegalStateException("Not loaded");
                long t0 = System.nanoTime();
                byte[] jpg = Base64.decode(img, Base64.DEFAULT);
                Bitmap bm = BitmapFactory.decodeByteArray(jpg, 0, jpg.length);
                if (bm == null) throw new IllegalStateException("Couldn't read the picture");
                if (bm.getWidth() != S || bm.getHeight() != S) {
                    Bitmap sc = Bitmap.createScaledBitmap(bm, S, S, true);
                    bm.recycle();
                    bm = sc;
                }
                bm.getPixels(pixels, 0, S, 0, 0, S, S);
                bm.recycle();
                fillInput(pixels);
                long t1 = System.nanoTime();
                float[] cand = run(session);
                long t2 = System.nanoTime();
                JSArray c = new JSArray();
                for (float v : cand) c.put((double) v);
                JSObject r = new JSObject();
                r.put("c", c);
                r.put("prep", Math.round((t1 - t0) / 1e6));
                r.put("ms", Math.round((t2 - t1) / 1e6));
                r.put("engine", engine);
                call.resolve(r);
            } catch (Throwable t) {
                Log.e(TAG, "detect failed", t);
                call.reject("Native detect failed: " + t.getMessage());
            }
        });
    }

    // ---------------------------------------------------------------------------------------

    private void init(String url) throws Exception {
        env = OrtEnvironment.getEnvironment();
        byte[] model = modelBytes(url);
        input = ByteBuffer.allocateDirect(3 * S * S * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();

        // Already timed on this phone? Use that engine straight away.
        SharedPreferences prefs = getContext().getSharedPreferences("mhvision", 0);
        String saved = prefs.getString("engine", null);
        if (saved != null) {
            try {
                OrtSession s = env.createSession(model, options(saved));
                inputName = s.getInputNames().iterator().next();
                session = s;
                engine = saved;
                engineMs = prefs.getFloat("ms", 0f);
                report = prefs.getString("report", "") + " (saved)";
                return;
            } catch (Throwable t) {
                Log.w(TAG, "saved engine " + saved + " failed; timing again", t);
                prefs.edit().remove("engine").apply();
            }
        }

        String[] kinds = { "chip", "cpu", "xnnpack" };
        OrtSession best = null;
        String bestKind = null;
        double bestMs = Double.MAX_VALUE;
        StringBuilder rep = new StringBuilder();

        for (String kind : kinds) {
            OrtSession s = null;
            try {
                s = env.createSession(model, options(kind));
                if (inputName == null) inputName = s.getInputNames().iterator().next();
                double ms = bench(s);
                rep.append(kind).append(' ').append(Math.round(ms)).append("ms; ");
                if (ms < bestMs) {
                    if (best != null) best.close();
                    best = s;
                    bestKind = kind;
                    bestMs = ms;
                    s = null;
                }
            } catch (Throwable t) {
                rep.append(kind).append(" failed; ");
                Log.w(TAG, "engine " + kind + " unavailable", t);
            } finally {
                if (s != null) {
                    try {
                        s.close();
                    } catch (Throwable ignored) {}
                }
            }
        }
        if (best == null) {
            // perhaps a damaged download: remove it so the next attempt fetches a fresh copy
            new File(new File(getContext().getFilesDir(), "models"), MODEL_FILE).delete();
            throw new IllegalStateException("No engine worked: " + rep);
        }
        session = best;
        engine = bestKind;
        engineMs = bestMs;
        report = rep.toString();
        prefs.edit().putString("engine", engine).putFloat("ms", (float) engineMs).putString("report", report).apply();
        Log.i(TAG, "Using " + engine + " (" + report + ")");
    }

    private OrtSession.SessionOptions options(String kind) throws Exception {
        int cores = Math.max(1, Math.min(4, Runtime.getRuntime().availableProcessors()));
        OrtSession.SessionOptions o = new OrtSession.SessionOptions();
        o.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        if (kind.equals("chip")) {
            o.addNnapi(EnumSet.of(NNAPIFlags.USE_FP16));
        } else if (kind.equals("cpu")) {
            o.setIntraOpNumThreads(cores);
        } else {
            o.setIntraOpNumThreads(1);
            o.addXnnpack(Collections.singletonMap("intra_op_num_threads", String.valueOf(cores)));
        }
        return o;
    }

    // The model, from the phone if it has been downloaded before, otherwise from the Madhouse site.
    private byte[] modelBytes(String url) throws Exception {
        File dir = new File(getContext().getFilesDir(), "models");
        if (!dir.exists()) dir.mkdirs();
        File f = new File(dir, MODEL_FILE);
        if (f.exists() && f.length() > 1000000) {
            try (InputStream in = new FileInputStream(f)) {
                return readAll(in, (int) f.length());
            }
        }
        if (url == null || url.isEmpty()) throw new IllegalStateException("No model address");
        File part = new File(dir, MODEL_FILE + ".part");
        HttpURLConnection c = (HttpURLConnection) new URL(url).openConnection();
        c.setConnectTimeout(15000);
        c.setReadTimeout(30000);
        try {
            if (c.getResponseCode() != 200) throw new IllegalStateException("Model download failed: " + c.getResponseCode());
            dlTotal = Math.max(0, c.getContentLengthLong());
            dlGot = 0;
            try (InputStream in = c.getInputStream(); FileOutputStream out = new FileOutputStream(part)) {
                byte[] b = new byte[1 << 16];
                int r;
                while ((r = in.read(b)) > 0) {
                    out.write(b, 0, r);
                    dlGot += r;
                }
            }
        } finally {
            c.disconnect();
        }
        if (part.length() < 1000000 || (dlTotal > 0 && part.length() != dlTotal)) {
            part.delete();
            throw new IllegalStateException("Model download incomplete");
        }
        if (f.exists()) f.delete();
        if (!part.renameTo(f)) throw new IllegalStateException("Couldn't save the model");
        try (InputStream in = new FileInputStream(f)) {
            return readAll(in, (int) f.length());
        }
    }

    // A few runs on a plain grey picture; the first run sets things up, so it is ignored.
    private double bench(OrtSession s) throws Exception {
        input.clear();
        for (int i = 0; i < 3 * S * S; i++) input.put(0.447f);
        input.rewind();
        run(s);
        double best = Double.MAX_VALUE;
        for (int k = 0; k < 2; k++) {
            long t = System.nanoTime();
            run(s);
            best = Math.min(best, (System.nanoTime() - t) / 1e6);
        }
        return best;
    }

    private void fillInput(int[] px) {
        int n = S * S;
        input.clear();
        for (int i = 0; i < n; i++) input.put(i, ((px[i] >> 16) & 0xff) / 255f);
        for (int i = 0; i < n; i++) input.put(n + i, ((px[i] >> 8) & 0xff) / 255f);
        for (int i = 0; i < n; i++) input.put(2 * n + i, (px[i] & 0xff) / 255f);
        input.rewind();
    }

    // Returns the candidates as a flat list: x, y, w, h, confidence, class, ...
    private float[] run(OrtSession s) throws Exception {
        input.rewind();
        try (OnnxTensor t = OnnxTensor.createTensor(env, input, new long[] { 1, 3, S, S });
             OrtSession.Result res = s.run(Collections.singletonMap(inputName, t))) {
            OnnxValue v = res.get(0);
            OnnxTensor out = (OnnxTensor) v;
            long[] dims = out.getInfo().getShape();
            int ch = (int) dims[1], N = (int) dims[2], C = ch - 4;
            FloatBuffer d = out.getFloatBuffer();
            float[] tmp = new float[64 * 6];
            int cnt = 0;
            for (int k = 0; k < N; k++) {
                float best = 0f;
                int bc = -1;
                for (int c = 0; c < C; c++) {
                    float q = d.get((4 + c) * N + k);
                    if (q > best) {
                        best = q;
                        bc = c;
                    }
                }
                if (best >= MIN_CONF) {
                    if ((cnt + 1) * 6 > tmp.length) {
                        float[] g = new float[tmp.length * 2];
                        System.arraycopy(tmp, 0, g, 0, tmp.length);
                        tmp = g;
                    }
                    int o = cnt * 6;
                    tmp[o] = d.get(k);
                    tmp[o + 1] = d.get(N + k);
                    tmp[o + 2] = d.get(2 * N + k);
                    tmp[o + 3] = d.get(3 * N + k);
                    tmp[o + 4] = best;
                    tmp[o + 5] = bc;
                    cnt++;
                }
            }
            float[] outArr = new float[cnt * 6];
            System.arraycopy(tmp, 0, outArr, 0, cnt * 6);
            return outArr;
        }
    }

    private static byte[] readAll(InputStream in, int sizeHint) throws Exception {
        ByteArrayOutputStream bo = new ByteArrayOutputStream(Math.max(sizeHint, 1 << 16));
        byte[] b = new byte[1 << 16];
        int r;
        while ((r = in.read(b)) > 0) bo.write(b, 0, r);
        return bo.toByteArray();
    }

    @Override
    protected void handleOnDestroy() {
        worker.shutdown();
        try {
            if (session != null) session.close();
        } catch (Throwable ignored) {}
        super.handleOnDestroy();
    }
}
