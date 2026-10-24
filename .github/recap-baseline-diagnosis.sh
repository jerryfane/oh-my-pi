#!/usr/bin/env bash
# Temporary evidence instrument for Jarvis356672; remove after retaining the receipts.
set -euo pipefail
base=d1872493752dd085173c38ecd38a1134dabf900f
head=440889d084e4f35e867da9119c695c45d16e8a87
out="$GITHUB_WORKSPACE/recap-baseline-evidence"
mkdir -p "$out" "$RUNNER_TEMP/recap-pinned-native"
printf 'base=%s\nhead=%s\nrunner=%s\nimage=%s\nbun=%s\n' "$base" "$head" "$RUNNER_NAME" "${ImageVersion:-unknown}" "$(bun --version)" | tee "$out/environment.txt"
df -Pk . | tee -a "$out/environment.txt"
url=https://registry.npmjs.org/@oh-my-pi/pi-natives-linux-x64/-/pi-natives-linux-x64-18.12.6.tgz
curl -fsSL "$url" -o "$RUNNER_TEMP/recap-native.tgz"
actual="$(openssl dgst -sha512 -binary "$RUNNER_TEMP/recap-native.tgz" | openssl base64 -A)"
expected='Dr03a4R2sU03/Qn5I9JgY8c3dnt7O2JNeV8FaFEJsiAlp8DKgfKfrsbVHDk4kqYgiNVfD4Zbyr20e3eFra6rSg=='
test "$actual" = "$expected"
printf 'url=%s\nintegrity=sha512-%s\nretained_run=40133190270\nretained_artifact=12575771453\nretained_zip_sha256=1ee5febba9e9b4b25eed44c8f5d54669003d285f4917e0c2615b8d1e2868a174\n' "$url" "$actual" | tee "$out/native-package.txt"
tar -xzf "$RUNNER_TEMP/recap-native.tgz" -C "$RUNNER_TEMP/recap-pinned-native"
sha256sum "$RUNNER_TEMP/recap-pinned-native/package/"*.node | tee "$out/pinned-native-sha256.txt"
cat > "$RUNNER_TEMP/recap-exports.cjs" <<'JS'
const addon = require(process.argv[2]);
if (!addon || Object.keys(addon).length === 0) throw Error('Native positive control failed');
console.log(JSON.stringify({file:process.argv[2], exports:Object.keys(addon).sort(), MacOSPowerAssertion:typeof addon.MacOSPowerAssertion}, null, 2));
JS
failed=0
run_case() {
  local directory="$1" label="$2"
  shift 2
  set +e
  (cd "$directory" && "$@") > "$out/$side-$label.log" 2>&1
  local status=$?
  set -e
  cat "$out/$side-$label.log"
  printf '%s\t%s\t%s\n' "$side" "$label" "$status" | tee -a "$out/status.tsv"
  if [ "$status" -ne 0 ]; then failed=1; fi
}
for side in base head; do
  ref="$base"
  if [ "$side" = head ]; then ref="$head"; fi
  root="$RUNNER_TEMP/recap-$side"
  git worktree add --detach "$root" "$ref"
  (cd "$root" && bun install --frozen-lockfile)
  mkdir -p "$root/packages/natives/native"
  cp "$RUNNER_TEMP/recap-pinned-native/package/"*.node "$root/packages/natives/native/"
  sha256sum "$root/packages/natives/native/"*.node | tee "$out/$side-native-sha256.txt"
  bun "$RUNNER_TEMP/recap-exports.cjs" "$root/packages/natives/native/pi_natives.linux-x64-baseline.node" > "$out/$side-exports.json"
  run_case "$root/packages/natives" natives bun test --parallel=2 --timeout=30000
  run_case "$root/packages/snapcompact" snapcompact bun test --parallel=2 --timeout=30000
  run_case "$root/packages/tui" sixel bun test test/image-render.test.ts test/slash-autocomplete-viewport.test.ts --parallel=2 --timeout=30000
  run_case "$root/packages/coding-agent" native-unit bun test test/core/block-replace.test.ts test/tools/ast-edit.test.ts test/tools/ast-grep.test.ts --parallel=1 --timeout=30000
  run_case "$root" install-methods bun run ci:test:install-methods
  sha256sum "$root/packages/natives/native/"*.node | tee "$out/$side-native-after-sha256.txt"
done
# A diagnostic comparison does not turn failing product checks green.
exit "$failed"
