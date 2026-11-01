import SwiftUI

/// Samples CIFAR-10 images on the device from weights trained by
/// examples/demo_cnn_rectified.cu, showing them turn from noise into images.
struct CIFARView: View {
    /// Starts sampling as soon as the view appears.
    var autostart = false
    /// Runs the CIFAR speed test in the console.
    var onSpeedTest: () -> Void

    @StateObject private var model = CIFARModel()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("Rectified flow on CIFAR-10").font(.title2.weight(.semibold))
                Text("A small UNet (examples/demo_cnn_rectified.cu) moves Gaussian noise along its "
                     + "learned velocity field from t = 0 to t = 1. The weights were trained with "
                     + "Juzhen; the sampling runs here, on this device's GPU.")
                    .font(.callout)

                if let missing = model.missing {
                    Text(missing).font(.callout).foregroundStyle(.red)
                }

                HStack {
                    Picker("Solver", selection: $model.heun) {
                        Text("Euler").tag(false)
                        Text("Heun").tag(true)
                    }
                    .pickerStyle(.menu)
                    Picker("Steps", selection: $model.steps) {
                        ForEach([10, 20, 50, 100], id: \.self) { Text("\($0) steps").tag($0) }
                    }
                    .pickerStyle(.menu)
                    Spacer()
                    Button("Generate images") { model.generate() }
                        .buttonStyle(.borderedProminent)
                }
                .disabled(model.running || model.missing != nil)

                Group {
                    if let image = model.frame?.image {
                        Image(decorative: image, scale: 1)
                            .interpolation(.none)
                            .resizable()
                    } else {
                        Rectangle().fill(Color(white: 0.1))
                    }
                }
                .aspectRatio(1, contentMode: .fit)
                .frame(maxWidth: 640)

                HStack(spacing: 10) {
                    Text("t = \(model.frame?.t ?? 0, specifier: "%.2f")").font(.footnote.monospaced())
                    Slider(value: Binding(get: { Double(model.index) },
                                          set: { model.index = Int($0.rounded()) }),
                           in: 0...Double(max(model.frames.count - 1, 1)), step: 1)
                        .disabled(model.running || model.frames.count < 2)
                }
                .frame(maxWidth: 640)

                Text(model.status).font(.footnote).foregroundStyle(.secondary)
            }
            .padding()
        }
        .toolbar { Button("Speed test", action: onSpeedTest).disabled(model.running) }
        .task { if autostart && model.frames.isEmpty { model.generate() } }
    }
}

final class CIFARModel: ObservableObject {
    struct Frame {
        let t: Double
        let image: CGImage
    }

    // FID-10k 41 at 100 Euler steps, 43 at 50
    @Published var steps = 100
    /// Heun's method: two network evaluations per step, more accurate per step
    @Published var heun = false
    @Published var index = 0
    @Published private(set) var frames: [Frame] = []
    @Published private(set) var running = false
    @Published private(set) var status = ""

    /// unet.ckpt from demo_cnn_rectified, copied into ios/JuzhenDemo/Weights.
    private static let checkpoint = Bundle.main.url(forResource: "unet", withExtension: "ckpt",
                                                    subdirectory: "Weights")

    var frame: Frame? { frames.indices.contains(index) ? frames[index] : nil }
    var missing: String? {
        Self.checkpoint == nil
            ? "No weights in the app: train with demo_cnn_rectified (it saves unet.ckpt) and "
              + "copy the file into ios/JuzhenDemo/Weights, then rebuild."
            : nil
    }

    func generate() {
        guard !running, let checkpoint = Self.checkpoint else { return }
        running = true
        frames = []
        index = 0
        status = "sampling..."

        let steps = self.steps
        let heun = self.heun
        // launch argument `-cifarCount n` samples fewer images, e.g. in the simulator
        let count = UserDefaults.standard.integer(forKey: "cifarCount") > 0
            ? UserDefaults.standard.integer(forKey: "cifarCount") : 64
        DispatchQueue.global(qos: .userInitiated).async {
            let start = Date()
            var epochs = 0
            let error = JuzhenRunner.cifarGenerate(checkpoint: checkpoint.path, count: count, steps: steps,
                                                   heun: heun, every: max(1, steps / 20), epochs: &epochs) {
                step, rgba, width, height in
                guard let image = Self.image(rgba: rgba, width: width, height: height) else { return }
                let frame = Frame(t: Double(step) / Double(steps), image: image)
                DispatchQueue.main.async {
                    self.frames.append(frame)
                    self.index = self.frames.count - 1
                }
            }
            let seconds = Date().timeIntervalSince(start)
            DispatchQueue.main.async {
                self.running = false
                self.status = error ?? "\(count) images, \(steps) \(heun ? "Heun" : "Euler") steps in "
                    + "\(String(format: "%.1f", seconds)) s on \(JuzhenRunner.gpuName()) "
                    + "(weights trained for \(epochs) epochs)"
            }
        }
    }

    private static func image(rgba: Data, width: Int, height: Int) -> CGImage? {
        guard let provider = CGDataProvider(data: rgba as CFData) else { return nil }
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }
}
