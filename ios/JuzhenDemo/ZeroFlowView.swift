import SwiftUI

/// Native version of examples/web/zeroflow.html. Training runs in Juzhen
/// (stream_zero_flow in examples/demo_zeroflow_web.cu); this view decodes its
/// NDJSON stream and draws the particles and the velocity field.
struct ZeroFlowView: View {
    /// Starts training as soon as the view appears.
    var autostart = false

    @StateObject private var model = ZeroFlowModel()
    @Environment(\.horizontalSizeClass) private var sizeClass

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                header
                if sizeClass == .regular {
                    HStack(alignment: .top, spacing: 28) {
                        controls.frame(width: 260)
                        stage
                    }
                } else {
                    stage
                    controls
                }
            }
            .padding()
        }
        .task { if autostart && model.frames.isEmpty { model.train() } }
        .onDisappear { model.cancel() }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("The Zero-Flow Phenomenon").font(.title2.weight(.semibold))
            Text("v\u{2081}\u{2215}\u{2082} \u{2261} 0  \u{21D4}  p_source = p_target \u{2014} Theorem 3.1 of "
                 + "Zero-Flow Encoders, Wang, Wang, Liu & Suzuki, ICML 2026")
                .font(.subheadline).italic().foregroundStyle(.secondary)
            Text("A rectified flow is trained with independent coupling between the two "
                 + "distributions you choose; particles are then transported from t = 0 to "
                 + "t = 1. The field vanishes everywhere at t = \u{00BD} if and only if the two "
                 + "distributions coincide. Pick the same dataset twice and watch everything "
                 + "freeze blue at the midpoint.")
                .font(.callout)
        }
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 14) {
            picker("source distribution", selection: $model.source)
            picker("target distribution", selection: $model.target)
            VStack(alignment: .leading, spacing: 4) {
                Text("training iterations").font(.caption).italic().foregroundStyle(.secondary)
                Picker("training iterations", selection: $model.iters) {
                    Text("2000 (fast)").tag(2000)
                    Text("4000").tag(4000)
                    Text("8000 (better field)").tag(8000)
                }
                .pickerStyle(.menu)
            }
            .disabled(model.training)

            Button { model.train() } label: {
                Text("Start: train & play").frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(model.training)

            Button { model.play() } label: {
                Text("Replay animation").frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .disabled(model.training || model.playing || model.frames.isEmpty)

            if model.training {
                ProgressView(value: Double(model.trainedIters), total: Double(max(model.totalIters, 1)))
            }

            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 4) {
                    // the swatches of the web page's legend
                    Text("particle speed:")
                    Circle().fill(Color(red: 0.30, green: 0.64, blue: 1)).frame(width: 9, height: 9)
                    Text("0 (still) \u{2192}")
                    Circle().fill(Color(red: 1, green: 0.82, blue: 0.30)).frame(width: 9, height: 9)
                    Text("\u{2192}")
                    Circle().fill(Color(red: 1, green: 0.30, blue: 0.30)).frame(width: 9, height: 9)
                    Text("fast")
                }
                Text("red arrows: learned velocity field v\u{209C}")
            }
            .font(.caption).foregroundStyle(.secondary)

            Text(model.status).font(.footnote).foregroundStyle(.secondary)
            if !model.timing.isEmpty {
                Text(model.timing).font(.footnote.monospaced()).foregroundStyle(.secondary)
            }
        }
    }

    private func picker(_ title: String, selection: Binding<ZeroFlowModel.Dataset>) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption).italic().foregroundStyle(.secondary)
            Picker(title, selection: selection) {
                ForEach(ZeroFlowModel.Dataset.allCases) { Text($0.title).tag($0) }
            }
            .pickerStyle(.menu)
            .disabled(model.training)
        }
    }

    private var stage: some View {
        VStack(spacing: 8) {
            FlowCanvas(model: model)
                .aspectRatio(1, contentMode: .fit)
                .overlay(alignment: .top) { if model.badgeVisible { badge } }
            HStack(spacing: 10) {
                Text("t = \(model.frame?.t ?? 0, specifier: "%.2f")")
                Slider(value: Binding(get: { Double(model.index) },
                                      set: { model.scrub(to: Int($0.rounded())) }),
                       in: 0...Double(max(model.frames.count - 1, 1)), step: 1)
                    .disabled(model.playing || model.frames.isEmpty)
                if let f = model.frame { Text("mean \u{2016}v\u{2016} = \(f.meanV, specifier: "%.3f")") }
            }
            .font(.footnote.monospaced())
        }
        .frame(maxWidth: 640)
    }

    private var badge: some View {
        VStack(spacing: 4) {
            Text("t = 0.50   mean \u{2016}v\u{2016} = \(model.frame?.meanV ?? 0, specifier: "%.3f")")
            if model.isZero {
                Text("ZERO VELOCITY").font(.title3.bold()).kerning(2).foregroundStyle(Color(red: 0.30, green: 0.64, blue: 1))
            } else {
                Text("v \u{2260} 0 \u{2192} p_source \u{2260} p_target").font(.headline)
                    .foregroundStyle(Color(red: 1, green: 0.61, blue: 0.30))
            }
        }
        .font(.footnote.monospaced().bold())
        .foregroundStyle(.white)
        .padding(.horizontal, 16).padding(.vertical, 8)
        .background(.black.opacity(0.65), in: RoundedRectangle(cornerRadius: 3))
        .padding(.top, 12)
    }
}

/// The particles (coloured by speed) and the velocity-field arrows of the
/// current frame, in the page's [-3.5, 3.5]^2 coordinates.
private struct FlowCanvas: View {
    @ObservedObject var model: ZeroFlowModel

    var body: some View {
        Canvas { ctx, size in
            let side = min(size.width, size.height)
            let lo: Float = -3.5, hi: Float = 3.5
            func px(_ x: Float) -> CGFloat { CGFloat((x - lo) / (hi - lo)) * side }
            func py(_ y: Float) -> CGFloat { side - CGFloat((y - lo) / (hi - lo)) * side }
            let unit = side / 560   // the web page draws on a 560 px canvas

            ctx.fill(Path(CGRect(x: 0, y: 0, width: side, height: side)),
                     with: .color(Color(red: 0x10 / 255, green: 0x14 / 255, blue: 0x18 / 255)))
            guard let f = model.frame else { return }

            var arrows = Path()
            for j in model.gridX.indices where 2 * j + 1 < f.vec.count {
                let u = f.vec[2 * j], v = f.vec[2 * j + 1]
                if hypot(u, v) < 0.02 { continue }
                let p0 = CGPoint(x: px(model.gridX[j]), y: py(model.gridY[j]))
                let p1 = CGPoint(x: p0.x + CGFloat(u) * 22 * unit, y: p0.y - CGFloat(v) * 22 * unit)
                arrows.move(to: p0)
                arrows.addLine(to: p1)
                let a = atan2(p1.y - p0.y, p1.x - p0.x)
                for da in [-0.4, 0.4] as [CGFloat] {   // arrowhead
                    arrows.move(to: p1)
                    arrows.addLine(to: CGPoint(x: p1.x - 4 * unit * cos(a + da),
                                               y: p1.y - 4 * unit * sin(a + da)))
                }
            }
            ctx.stroke(arrows, with: .color(Color(red: 1, green: 80 / 255, blue: 80 / 255).opacity(0.55)),
                       lineWidth: 1)

            let r = 3.2 * unit
            for j in stride(from: 0, to: f.pts.count - 2, by: 3) {
                let c = CGPoint(x: px(f.pts[j]), y: py(f.pts[j + 1]))
                ctx.fill(Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r)),
                         with: .color(model.speedColor(f.pts[j + 2])))
            }
        }
        .border(Color(red: 0x8f / 255, green: 0x8a / 255, blue: 0x7e / 255))
    }
}

final class ZeroFlowModel: ObservableObject {
    enum Dataset: String, CaseIterable, Identifiable {
        case mixture, gaussian, moons, ring, spiral
        var id: String { rawValue }
        var title: String {
            switch self {
            case .mixture: "gaussian mixture"
            case .gaussian: "gaussian"
            case .moons: "two moons"
            case .ring: "ring"
            case .spiral: "spiral"
            }
        }
    }

    struct Frame {
        let t: Float
        let meanV: Float
        let pts: [Float]   // x, y, speed per particle
        let vec: [Float]   // u, v per grid point
    }

    @Published var source = Dataset.mixture
    @Published var target = Dataset.mixture
    @Published var iters = 4000
    @Published private(set) var training = false
    @Published private(set) var trainedIters = 0
    @Published private(set) var totalIters = 0
    @Published private(set) var status = ""
    @Published private(set) var timing = ""
    @Published private(set) var gridX: [Float] = []
    @Published private(set) var gridY: [Float] = []
    @Published private(set) var frames: [Frame] = []
    @Published private(set) var index = 0
    @Published private(set) var playing = false
    @Published private(set) var badgeVisible = false
    /// Speed that maps to full red: the 95th-percentile particle speed.
    @Published private(set) var smax: Float = 1

    private var run: Cancellation?
    private var playback: Task<Void, Never>?

    var frame: Frame? { frames.indices.contains(index) ? frames[index] : nil }
    var isZero: Bool { (frame?.meanV ?? 0) < 0.12 * smax }

    /// Blue (still) through yellow to red (fast), as in the web page.
    func speedColor(_ s: Float) -> Color {
        let r = Double(min(1, s / smax))
        let l = (55 + 10 * (1 - r)) / 100, sl = 0.9   // hsl(212(1-r), 90%, l)
        let v = l + sl * min(l, 1 - l)
        return Color(hue: 212 * (1 - r) / 360, saturation: v == 0 ? 0 : 2 * (1 - l / v), brightness: v)
    }

    func train() {
        guard !training else { return }
        cancel()
        let token = Cancellation()
        run = token
        training = true
        frames = []
        trainedIters = 0
        totalIters = iters
        status = "training velocity field..."
        timing = ""

        let (src, tgt, n) = (source.rawValue, target.rawValue, iters)
        DispatchQueue.global(qos: .userInitiated).async {
            let start = Date()
            let decoder = JSONDecoder()
            var received: [Frame] = []
            var grid: Message?
            let ok = JuzhenRunner.zeroFlow(source: src, target: tgt, iters: n) { line in
                if token.isCancelled { return false }
                guard let msg = try? decoder.decode(Message.self, from: line) else { return true }
                switch msg.phase {
                case "train":
                    let (iter, total, loss) = (msg.iter ?? 0, msg.total ?? n, msg.loss ?? 0)
                    DispatchQueue.main.async {
                        guard self.run === token else { return }
                        self.trainedIters = iter + 1
                        self.status = "training... iter \(iter)/\(total), loss \(String(format: "%.3f", loss))"
                    }
                case "grid":
                    grid = msg
                case "frame":
                    received.append(Frame(t: msg.t ?? 0, meanV: msg.meanV ?? 0,
                                          pts: msg.pts ?? [], vec: msg.vec ?? []))
                default:
                    break
                }
                return true
            }
            let seconds = Date().timeIntervalSince(start)

            DispatchQueue.main.async {
                guard self.run === token, !token.isCancelled else { return }
                self.training = false
                guard ok, !received.isEmpty else {
                    self.status = "training failed"
                    return
                }
                let speeds = received.flatMap { f in stride(from: 2, to: f.pts.count, by: 3).map { f.pts[$0] } }
                    .sorted()
                self.smax = max(0.3, speeds.isEmpty ? 1 : speeds[Int(Double(speeds.count) * 0.95)])
                self.gridX = grid?.x ?? []
                self.gridY = grid?.y ?? []
                self.frames = received
                self.timing = "\(n) iterations in \(String(format: "%.1f", seconds)) s on \(JuzhenRunner.gpuName())"
                self.play()
            }
        }
    }

    /// Plays t = 0 ... 1, pausing at t = 0.5 to show the mean speed there.
    func play() {
        guard !frames.isEmpty else { return }
        playback?.cancel()
        playing = true
        badgeVisible = false
        status = "playing..."
        let mid = Int((Double(frames.count - 1) / 2).rounded())
        playback = Task { @MainActor in
            for k in frames.indices {
                index = k
                if k == mid {
                    badgeVisible = true
                    try? await Task.sleep(for: .seconds(2.2))
                    badgeVisible = false
                } else {
                    try? await Task.sleep(for: .milliseconds(90))
                }
                if Task.isCancelled { return }
            }
            playing = false
            status = "done \u{2014} scrub the timeline or replay."
        }
    }

    func scrub(to k: Int) {
        guard !playing, frames.indices.contains(k) else { return }
        index = k
        badgeVisible = abs(frames[k].t - 0.5) < 1e-6
    }

    /// Stops playback and any training in progress (Juzhen stops at its next line).
    func cancel() {
        run?.cancel()
        playback?.cancel()
        playing = false
        badgeVisible = false
        if training {
            training = false
            status = ""
        }
    }

    /// One NDJSON line of stream_zero_flow.
    private struct Message: Decodable {
        let phase: String
        var iter: Int?, total: Int?, loss: Float?
        var x: [Float]?, y: [Float]?
        var t: Float?, meanV: Float?, pts: [Float]?, vec: [Float]?

        enum CodingKeys: String, CodingKey {
            case phase, iter, total, loss, x, y, t, pts, vec
            case meanV = "mean_v"
        }
    }

    private final class Cancellation {
        private let lock = NSLock()
        private var cancelled = false
        var isCancelled: Bool { lock.withLock { cancelled } }
        func cancel() { lock.withLock { cancelled = true } }
    }
}
