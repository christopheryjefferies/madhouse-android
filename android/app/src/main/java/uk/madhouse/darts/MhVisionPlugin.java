package uk.madhouse.darts;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.providers.NNAPIFlags;
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
import java.io.InputStream;
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
    private final int[] pixels = new int[S * S];

    @PluginMethod
    public void load(final PluginCall call) {
        worker.execute(() -> {
            try {
                if (session == null) init();
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

    @PluginMethod
    public void detect(final PluginCall call) {
        final String img = call.getString("img");
        if (img == null) {
            call.reject("No picture");
            return;
        }
        worker.execute(() -> {
            try {
                if (session == null) init();
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

    private void init() throws Exception {
        env = OrtEnvironment.getEnvironment();
        byte[] model = readAsset("darts-sense-v1.onnx");
        input = ByteBuffer.allocateDirect(3 * S * S * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();

        int cores = Math.max(1, Math.min(4, Runtime.getRuntime().availableProcessors()));
        String[] kinds = { "chip", "cpu", "xnnpack" };
        OrtSession best = null;
        String bestKind = null;
        double bestMs = Double.MAX_VALUE;
        StringBuilder rep = new StringBuilder();

        for (String kind : kinds) {
            OrtSession s = null;
            try {
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
                s = env.createSession(model, o);
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
        if (best == null) throw new IllegalStateException("No engine worked: " + rep);
        session = best;
        engine = bestKind;
        engineMs = bestMs;
        report = rep.toString();
        Log.i(TAG, "Using " + engine + " (" + report + ")");
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

    private byte[] readAsset(String name) throws Exception {
        try (InputStream in = getContext().getAssets().open(name)) {
            ByteArrayOutputStream bo = new ByteArrayOutputStream(13 << 20);
            byte[] b = new byte[1 << 16];
            int r;
            while ((r = in.read(b)) > 0) bo.write(b, 0, r);
            return bo.toByteArray();
        }
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
