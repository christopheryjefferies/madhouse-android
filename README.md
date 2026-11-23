# Madhouse — Android app

The Android app is a thin shell around the live Madhouse site
(https://darts.christopheryjefferies.workers.dev). Game changes go live by deploying the site as
usual; the app picks them up straight away, exactly like the web version. A new APK is only
needed when the app shell itself changes (icon, permissions, native features).

## Building
Every push to `main` builds a signed `madhouse.apk` (for direct download) and `madhouse.aab`
(for the Play Store) on GitHub Actions. Download them from the run's **Artifacts** section.
You can also start a build by hand: Actions → Build Madhouse app → Run workflow.

## Secrets (Settings → Secrets and variables → Actions)
- `MH_KEYSTORE_B64` — the signing key, base64-encoded
- `MH_STORE_PASSWORD` — its password

Keep a copy of the key and password somewhere safe. Updates to the app must be signed with the
same key, or people have to uninstall and reinstall.

## What the shell adds
- Camera permission for camera scoring
- Keeps the screen awake while open
- Android Back button closes pop-ups and uses each screen's own Back button; it never quits a game
- An offline page if the app is opened for the very first time with no internet
- Native dart detection: on first use it times the phone's AI chip (Android neural networks API)
  and its main processor, and keeps the fastest. Madhouse sends it the same 800x800 picture it
  would otherwise score in the browser; if the native engine fails, Madhouse falls back to the
  browser engine automatically. The model (`android/app/src/main/assets/darts-sense-v1.onnx`) is
  Dart Sense v1, simplified with onnxsim (identical output).

Camera scoring uses the Dart Sense model by Ben Willshaw (CC BY-NC 4.0), so Madhouse stays free.

