import SwiftUI

@main
struct JuzhenDemoApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

/// What the running example has written to std::cout so far.
final class Console: ObservableObject {
    @Published var text = ""
    @Published var running = false

    func run(_ example: JuzhenExample) {
        guard !running else { return }
        running = true
        text = ""
        DispatchQueue.global(qos: .userInitiated).async {
            let start = Date()
            let ret = JuzhenRunner.run(example) { chunk in
                DispatchQueue.main.async { self.text += Console.stripANSI(chunk) }
            }
            let seconds = Date().timeIntervalSince(start)
            DispatchQueue.main.async {
                self.text += String(format: "\n[exit code %d, %.1f s]\n", ret, seconds)
                self.running = false
            }
        }
    }

    // demo_gemm.cu colours its result lines for a terminal.
    private static func stripANSI(_ s: String) -> String {
        s.replacingOccurrences(of: "\u{1B}\\[[0-9;]*m", with: "", options: .regularExpression)
    }
}

struct ContentView: View {
    @StateObject private var console = Console()
    @State private var showZeroFlow = false
    @State private var autostartZeroFlow = false
    @State private var showCIFAR = false
    @State private var autostartCIFAR = false

    var body: some View {
        NavigationStack {
            ScrollViewReader { proxy in
                ScrollView {
                    Text(console.text.isEmpty
                         ? "GPU: \(JuzhenRunner.gpuName())\n\nPick an example below."
                         : console.text)
                        .font(.system(size: 12, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                    Color.clear.frame(height: 1).id("end")
                }
                .onChange(of: console.text) { proxy.scrollTo("end", anchor: .bottom) }
            }
            .safeAreaInset(edge: .bottom) {
                HStack {
                    Button("Hello") { console.run(.helloWorld) }
                    Button("GEMM") { console.run(.GEMM) }
                    Button("CIFAR") { showCIFAR = true }
                    Button("Zero-Flow") { showZeroFlow = true }
                }
                .buttonStyle(.borderedProminent)
                .disabled(console.running)
                .padding()
                .frame(maxWidth: .infinity)
                .background(.bar)
            }
            .navigationTitle("Juzhen")
            .toolbar { if console.running { ProgressView() } }
            .navigationDestination(isPresented: $showZeroFlow) {
                ZeroFlowView(autostart: autostartZeroFlow)
                    .navigationTitle("Zero-Flow")
                    .navigationBarTitleDisplayMode(.inline)
            }
            .navigationDestination(isPresented: $showCIFAR) {
                CIFARView(autostart: autostartCIFAR) {
                    showCIFAR = false
                    console.run(.cifarSpeed)
                }
                .navigationTitle("CIFAR-10")
                .navigationBarTitleDisplayMode(.inline)
            }
        }
        .task {
            // Launch argument for running without tapping, e.g. `-autorun gemm`.
            switch UserDefaults.standard.string(forKey: "autorun") {
            case "helloworld": console.run(.helloWorld)
            case "gemm": console.run(.GEMM)
            case "cifar": console.run(.cifarSpeed)
            case "cifargen":
                autostartCIFAR = true
                showCIFAR = true
            case "zeroflow":
                autostartZeroFlow = true
                showZeroFlow = true
            default: break
            }
        }
    }
}
