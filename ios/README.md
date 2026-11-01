# Juzhen on iPhone / iPad

A minimal SwiftUI app that runs `examples/helloworld.cu` and `examples/demo_gemm.cu`
unchanged on the Metal backend and shows their `std::cout` output on screen, plus the
Zero-Flow demo (`examples/demo_zeroflow_web.cu`), which trains on the device and shows
the particles in a native SwiftUI port of `examples/web/zeroflow.html`.

## Run it

1. Xcode > Settings > Components: install the iOS platform (needed for devices and the simulator).
2. Open `ios/JuzhenDemo.xcodeproj`.
3. Target JuzhenDemo > Signing & Capabilities: pick your Team (a free Apple ID works).
   If the bundle identifier is taken, change it to anything unique.
4. Select your iPhone/iPad as the run destination and press ⌘R.

## How it is wired

- `JuzhenRunner.mm` sets up Metal once, releases the memory pools after each run as
  `main()` in `cpp/launcher.cu` does, and forwards `std::cout` to the UI. Runs take
  turns, since the matrix library is not thread-safe.
- Zero-Flow has no HTTP server on the device: `ZeroFlowView.swift` calls
  `stream_zero_flow()` from the demo (the code behind the Mac server's `POST /train`),
  decodes its NDJSON lines, and draws the frames with the web page's colours, arrow
  scale and t = 0.5 pause.
- Each example's `compute()` is renamed with a per-file flag
  (`-Dcompute=juzhen_helloworld`, `-Dcompute=juzhen_demo_gemm`) so both link into one app.
- `cpp/metal/shaders.metal` is compiled by Xcode into the app's `default.metallib`.
- `cpp/metal/MPSWrapper.mm` uses manual retain/release, so it is built with `-fno-objc-arc`.
- C++ is built at `-O3` in Debug too, so ⌘R timings are meaningful.
- The launch argument `-autorun helloworld|gemm|zeroflow` runs an example without
  tapping (Zero-Flow then starts training with the page's default settings).

To add another example, add its `.cu` file to the target (file type C++ source) with
`-Dcompute=<new name>`, then add a case to `JuzhenExample` and a button.
