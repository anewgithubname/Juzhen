#!/bin/bash
# Build the single-file offline zero-flow demo:
#   examples/web/zeroflow_standalone.html
# Requires Emscripten (https://emscripten.org): source ~/emsdk/emsdk_env.sh
set -e
cd "$(dirname "$0")"

emcc zeroflow_wasm.cpp \
    -O3 -msimd128 -ffast-math \
    -s MODULARIZE=1 -s EXPORT_NAME=createZF \
    -s SINGLE_FILE=1 \
    -s EXPORTED_RUNTIME_METHODS=HEAPF32 \
    -s ALLOW_MEMORY_GROWTH=1 \
    -o zf.js

# splice the (base64-embedded-wasm) JS module into the HTML shell
python3 - <<'EOF'
shell = open('shell.html').read()
js = open('zf.js').read()
out = shell.replace('<!--ZF_JS-->', '<script>\n' + js + '\n</script>')
open('../web/zeroflow_standalone.html', 'w').write(out)
print('wrote ../web/zeroflow_standalone.html (%.1f KB)' % (len(out) / 1024))
EOF
rm -f zf.js
