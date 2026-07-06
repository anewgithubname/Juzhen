#!/bin/bash
# Build the single-file offline zero-flow demo:
#   examples/web/zeroflow_combined_standalone.html
# Tabbed page with both the unconditional (Theorem 3.1) and the conditional
# (Theorem 3.4) zero-flow visualizations, light/dark themes.
# Requires Emscripten (https://emscripten.org): source ~/emsdk/emsdk_env.sh
set -e
cd "$(dirname "$0")"

compile() {  # $1 cpp  $2 export-name  $3 output-js
    emcc "$1" \
        -O3 -msimd128 -ffast-math \
        -s MODULARIZE=1 -s EXPORT_NAME="$2" \
        -s SINGLE_FILE=1 \
        -s EXPORTED_RUNTIME_METHODS=HEAPF32 \
        -s ALLOW_MEMORY_GROWTH=1 \
        -o "$3"
}

compile zeroflow_wasm.cpp      createZF  zf.js
compile zeroflow_cond_wasm.cpp createZFC zfc.js

# splice the (base64-embedded-wasm) JS modules into the HTML shell
python3 - <<'EOF'
out = open('shell_combined.html').read()
for placeholder, js_path in [('<!--ZF_JS-->', 'zf.js'), ('<!--ZFC_JS-->', 'zfc.js')]:
    js = open(js_path).read()
    out = out.replace(placeholder, '<script>\n' + js + '\n</script>')
path = '../web/zeroflow_combined_standalone.html'
open(path, 'w').write(out)
print('wrote %s (%.1f KB)' % (path, len(out) / 1024))
EOF
rm -f zf.js zfc.js
