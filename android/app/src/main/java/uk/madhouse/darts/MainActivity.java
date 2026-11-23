package uk.madhouse.darts;

import android.os.Bundle;
import android.view.WindowManager;
import android.webkit.WebView;
import androidx.activity.OnBackPressedCallback;
import com.getcapacitor.BridgeActivity;

public class MainActivity extends BridgeActivity {

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Keep the screen awake while Madhouse is open (scoring, camera, games in progress).
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        // Android back button / gesture: let Madhouse handle it (close a pop-up, press the
        // screen's own Back button). If Madhouse has nothing to go back to, tuck the app away
        // rather than closing it, so a game in progress is kept.
        getOnBackPressedDispatcher().addCallback(this, new OnBackPressedCallback(true) {
            @Override
            public void handleOnBackPressed() {
                WebView wv = (bridge != null) ? bridge.getWebView() : null;
                if (wv == null) {
                    moveTaskToBack(true);
                    return;
                }
                wv.evaluateJavascript(
                    "(function(){try{return !!(window.mhBack&&window.mhBack());}catch(e){return false;}})()",
                    value -> {
                        if (!"true".equals(value)) moveTaskToBack(true);
                    }
                );
            }
        });
    }
}
