#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NDK_DEFAULT="/opt/homebrew/share/android-commandlinetools/ndk/28.2.13676358"
NDK="${ANDROID_NDK_HOME:-$NDK_DEFAULT}"
API="${ANDROID_API:-35}"
TOOLCHAIN="$NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin"

TARGETS=(
  "arm64-v8a:aarch64-linux-android:aarch64-linux-android"
  "armeabi-v7a:armv7-linux-androideabi:armv7a-linux-androideabi"
  "x86_64:x86_64-linux-android:x86_64-linux-android"
)
for entry in "${TARGETS[@]}"; do
  IFS=":" read -r abi target clang_prefix <<< "$entry"
  rustup target add "$target" >/dev/null 2>&1 || true
  env_target="$(echo "$target" | tr 'a-z-' 'A-Z_')"
  linker="$TOOLCHAIN/${clang_prefix}${API}-clang"
  export CARGO_TARGET_${env_target}_LINKER="$linker"
  cargo build --release --target "$target"
  mkdir -p "$ROOT/dist/android/$abi"
  cp "$ROOT/target/$target/release/libccs_algorithm.so" "$ROOT/dist/android/$abi/"
done

APPLE_TARGETS=(aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios)
for target in "${APPLE_TARGETS[@]}"; do
  rustup target add "$target" >/dev/null 2>&1 || true
  cargo build --release --target "$target"
done
mkdir -p "$ROOT/dist/apple/ios/device" "$ROOT/dist/apple/ios/sim"
cp "$ROOT/target/aarch64-apple-ios/release/libccs_algorithm.a" \
  "$ROOT/dist/apple/ios/device/libccs_algorithm.a"
lipo -create \
  "$ROOT/target/aarch64-apple-ios-sim/release/libccs_algorithm.a" \
  "$ROOT/target/x86_64-apple-ios/release/libccs_algorithm.a" \
  -output "$ROOT/dist/apple/ios/sim/libccs_algorithm_ios_sim.a"

echo "CCS mobile libraries assembled under dist/android and dist/apple."
