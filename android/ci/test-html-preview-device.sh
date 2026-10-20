#!/usr/bin/env bash
set -euo pipefail
preview_api="${HTML_PREVIEW_API:-37.0}"
report_dir="$RUNNER_TEMP/html-preview-device/$preview_api"
export ANDROID_USER_HOME="$RUNNER_TEMP/html-preview-android-$preview_api"
export ANDROID_EMULATOR_HOME="$ANDROID_USER_HOME"
export ANDROID_AVD_HOME="$ANDROID_USER_HOME/avd"
export PATH="$ANDROID_HOME/platform-tools:$PATH"
mkdir -p "$report_dir" "$ANDROID_AVD_HOME"
image="system-images;android-$preview_api;google_apis;x86_64"
timeout 300 "$ANDROID_HOME/cmdline-tools/latest/bin/sdkmanager" emulator "$image" > "$report_dir/sdk.log" 2>&1
sudo chmod 666 /dev/kvm
printf 'no\n' | "$ANDROID_HOME/cmdline-tools/latest/bin/avdmanager" create avd --force -n html-preview --device pixel --path "$ANDROID_AVD_HOME/html-preview.avd" -k "$image" > "$report_dir/avd.log" 2>&1
sed -i 's/^disk.dataPartition.size=.*/disk.dataPartition.size=4G/;s/^hw.ramSize=.*/hw.ramSize=4096M/;s/^vm.heapSize=.*/vm.heapSize=512M/' "$ANDROID_AVD_HOME/html-preview.avd/config.ini"
adb kill-server || true
adb start-server
cp "$ANDROID_AVD_HOME/html-preview.avd/config.ini" "$report_dir/device-config.ini"
"$ANDROID_HOME/emulator/emulator" -avd html-preview -memory 4096 -cores 2 -skin 480x960 -no-metrics -show-kernel -feature -Vulkan -no-window -no-audio -no-boot-anim -no-snapshot -gpu swiftshader > "$report_dir/emulator.log" 2>&1 &
emulator_pid=$!
trap 'kill "$emulator_pid" 2>/dev/null || true; wait "$emulator_pid" 2>/dev/null || true; adb kill-server || true' EXIT
if ! ./gradlew --no-daemon assembleDebug assembleDebugAndroidTest > "$report_dir/build.log" 2>&1; then
  tail -n 65 "$report_dir/build.log"
  exit 1
fi
if ! timeout 300 adb wait-for-device; then
  tail -n 45 "$report_dir/emulator.log"
  exit 1
fi
ready=0
for attempt in $(seq 1 180); do
  if [[ "$(adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" == "1" ]] &&
      adb shell pm path android 2>/dev/null | grep -q '^package:' &&
      [[ "$(adb shell am get-started-user-state 0 2>/dev/null | tr -d '\r')" == "RUNNING_UNLOCKED" ]]; then
    ready=$((ready + 1))
    if [[ "$ready" -ge 5 ]]; then break; fi
  else
    ready=0
  fi
  sleep 2
done
if [[ "$ready" -lt 5 ]]; then
  adb logcat -d -b crash > "$report_dir/boot-crashes.log" || true
  tail -n 45 "$report_dir/emulator.log"
  exit 1
fi
adb shell input keyevent 82
adb shell df /data > "$report_dir/data-space.log"
set +e
./gradlew --no-daemon connectedDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=com.minashin1120.aiplayground.ui.HtmlPreviewDeviceTest > "$report_dir/test.log" 2>&1
status=$?
adb pull /sdcard/Pictures/html-preview-device "$report_dir/paint" || true
adb exec-out screencap -p > "$report_dir/screen.png"
adb logcat -d | grep -E 'chromium|HTML_PREVIEW|TestRunner|AndroidRuntime' | tail -n 100 > "$report_dir/webview.log"
if [[ $status -ne 0 ]]; then
  tail -n 65 "$report_dir/test.log"
  cat "$report_dir/webview.log"
else
  tail -n 12 "$report_dir/test.log"
fi
exit "$status"
