#!/usr/bin/env bash
set -euo pipefail
report_dir="$RUNNER_TEMP/html-preview-device"
mkdir -p "$report_dir"
timeout 300 "$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager" "emulator" "system-images;android-35;google_apis;x86_64" > "$report_dir/sdk.log" 2>&1
sudo chmod 666 /dev/kvm
printf 'no\n' | "$ANDROID_HOME/cmdline-tools/latest/bin/avdmanager" create avd --force -n html-preview -k 'system-images;android-35;google_apis;x86_64' > "$report_dir/avd.log" 2>&1
"$ANDROID_HOME/emulator/emulator" -avd html-preview -no-window -no-audio -no-boot-anim -no-snapshot -gpu swiftshader > "$report_dir/emulator.log" 2>&1 &
emulator_pid=$!
trap 'kill "$emulator_pid" 2>/dev/null || true' EXIT
export PATH="$ANDROID_HOME/platform-tools:$PATH"
if ! timeout 180 adb wait-for-device; then
  tail -n 60 "$report_dir/emulator.log"
  exit 1
fi
for attempt in $(seq 1 120); do
  if [[ "$(adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" == "1" ]]; then break; fi
  sleep 2
done
[[ "$(adb shell getprop sys.boot_completed | tr -d '\r')" == "1" ]]
adb shell input keyevent 82
set +e
./gradlew --no-daemon connectedDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=com.minashin1120.aiplayground.ui.HtmlPreviewDeviceTest > "$report_dir/test.log" 2>&1
status=$?
adb pull /sdcard/Android/data/com.minashin1120.aiplayground/files/html-preview-attempt.png "$report_dir/attempt.png" || true
adb exec-out screencap -p > "$report_dir/screen.png"
adb logcat -d | grep -E 'chromium|HTML_PREVIEW|TestRunner|AndroidRuntime' | tail -n 100 > "$report_dir/webview.log"
if [[ $status -ne 0 ]]; then
  tail -n 65 "$report_dir/test.log"
  cat "$report_dir/webview.log"
else
  tail -n 12 "$report_dir/test.log"
fi
exit "$status"
