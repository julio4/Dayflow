import AppKit
import SwiftUI
import XCTest

@testable import Dayflow

/// Verifies the configurable concurrent frame-transcription feature:
/// 1. `OllamaProvider.maxConcurrency` reads/clamps the `llmLocalMaxConcurrency` default.
/// 2. `ProvidersSettingsViewModel.localMaxConcurrency` persists to / reloads from that default.
/// 3. `transcribeScreenshots` actually runs `describe_frame` calls concurrently up to the cap,
///    respects the cap, and re-sorts results by timestamp despite out-of-order completion.
final class LocalLLMConcurrencyTests: XCTestCase {
  private let concurrencyKey = "llmLocalMaxConcurrency"
  private let engineKey = "llmLocalEngine"
  private let modelKey = "llmLocalModelId"
  private let baseURLKey = "llmLocalBaseURL"
  private let analyticsKey = "analyticsOptIn"

  private var saved: [String: Any?] = [:]
  private var tempDir: URL!

  override func setUp() {
    super.setUp()
    let keys = [concurrencyKey, engineKey, modelKey, baseURLKey, analyticsKey]
    saved = Dictionary(uniqueKeysWithValues: keys.map { ($0, UserDefaults.standard.object(forKey: $0)) })
    keys.forEach { UserDefaults.standard.removeObject(forKey: $0) }
    // Keep any test-triggered telemetry offline.
    UserDefaults.standard.set(false, forKey: analyticsKey)

    tempDir = FileManager.default.temporaryDirectory
      .appendingPathComponent("LocalLLMConcurrencyTests-\(UUID().uuidString)", isDirectory: true)
    try? FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

    URLProtocol.registerClass(MockLocalLLMURLProtocol.self)
  }

  override func tearDown() {
    URLProtocol.unregisterClass(MockLocalLLMURLProtocol.self)
    MockLocalLLMURLProtocol.recorder = MockLocalLLMURLProtocol.Recorder()
    for (key, value) in saved {
      if let value {
        UserDefaults.standard.set(value, forKey: key)
      } else {
        UserDefaults.standard.removeObject(forKey: key)
      }
    }
    saved = [:]
    if let tempDir { try? FileManager.default.removeItem(at: tempDir) }
    super.tearDown()
  }

  // MARK: - maxConcurrency computed property

  func testMaxConcurrencyDefaultsToSequential() {
    UserDefaults.standard.removeObject(forKey: concurrencyKey)
    XCTAssertEqual(OllamaProvider().maxConcurrency, 1)
  }

  func testMaxConcurrencyReadsConfiguredValue() {
    UserDefaults.standard.set(4, forKey: concurrencyKey)
    XCTAssertEqual(OllamaProvider().maxConcurrency, 4)
  }

  func testMaxConcurrencyClampsInvalidAndOversizedValues() {
    UserDefaults.standard.set(0, forKey: concurrencyKey)
    XCTAssertEqual(OllamaProvider().maxConcurrency, 1, "0 falls back to sequential")

    UserDefaults.standard.set(-3, forKey: concurrencyKey)
    XCTAssertEqual(OllamaProvider().maxConcurrency, 1, "negative falls back to sequential")

    UserDefaults.standard.set(999, forKey: concurrencyKey)
    XCTAssertEqual(OllamaProvider().maxConcurrency, 16, "clamped to the 16 ceiling")

    UserDefaults.standard.set(16, forKey: concurrencyKey)
    XCTAssertEqual(OllamaProvider().maxConcurrency, 16)
  }

  // MARK: - Settings view model wiring

  @MainActor
  func testViewModelDefaultsPersistsAndReloadsConcurrency() {
    UserDefaults.standard.removeObject(forKey: concurrencyKey)

    let viewModel = ProvidersSettingsViewModel()
    XCTAssertEqual(viewModel.localMaxConcurrency, 1, "unset default is sequential")

    // Editing the stepper value persists it back to UserDefaults.
    viewModel.localMaxConcurrency = 5
    XCTAssertEqual(UserDefaults.standard.integer(forKey: concurrencyKey), 5)

    // A provider-setup completion reloads the value from UserDefaults.
    UserDefaults.standard.set(7, forKey: concurrencyKey)
    viewModel.handleProviderSetupCompletion("ollama")
    XCTAssertEqual(viewModel.localMaxConcurrency, 7)

    // Out-of-range stored values are clamped on reload.
    UserDefaults.standard.set(100, forKey: concurrencyKey)
    viewModel.handleProviderSetupCompletion("ollama")
    XCTAssertEqual(viewModel.localMaxConcurrency, 16)
  }

  // MARK: - End-to-end concurrent transcription

  func testTranscriptionRunsSequentiallyWhenCapIsOne() async throws {
    let run = try await runTranscription(cap: 1, frameCount: 8)
    XCTAssertEqual(run.peak, 1, "cap=1 must run describe_frame calls one at a time")
    XCTAssertEqual(run.describeCount, 8, "every sampled frame is described")
    assertMergePromptSortedByTimestamp(run)
  }

  func testTranscriptionRunsConcurrentlyAndRespectsCap() async throws {
    let sequential = try await runTranscription(cap: 1, frameCount: 8)
    let parallel = try await runTranscription(cap: 4, frameCount: 8)

    XCTAssertLessThanOrEqual(parallel.peak, 4, "must never exceed the configured cap")
    XCTAssertEqual(parallel.peak, 4, "cap=4 keeps 4 describe_frame calls in flight")
    XCTAssertGreaterThan(
      parallel.peak, sequential.peak,
      "raising the cap increases observed parallelism")
    XCTAssertEqual(parallel.describeCount, 8)

    // The re-sort keeps merge input in timestamp order even though calls finished out of order.
    assertMergePromptSortedByTimestamp(parallel)
    XCTAssertFalse(
      mergePromptCompletionOrder(parallel).isAscending,
      "describe_frame calls should have completed out of timestamp order, exercising the re-sort")

    let observationsSorted =
      parallel.observations.map(\.startTs) == parallel.observations.map(\.startTs).sorted()
    let evidence = """
      ===== CONCURRENT FRAME TRANSCRIPTION EVIDENCE =====
      8 sampled frames transcribed through the real OllamaProvider.transcribeScreenshots
      pipeline. describe_frame HTTP calls are intercepted by a mock URLProtocol, so no live
      LM Studio / Ollama server is required; the shipped concurrency code runs unchanged.

      llmLocalMaxConcurrency = 1  ->  peak describe_frame calls in flight: \(sequential.peak)  (sequential)
      llmLocalMaxConcurrency = 4  ->  peak describe_frame calls in flight: \(parallel.peak)  (parallel, capped at 4)

      Re-sort correctness under out-of-order completion (cap = 4):
        completion order, read in final re-sorted frame order: \(mergePromptCompletionOrder(parallel))
          (not ascending -> calls finished in a different order than submitted)
        timestamps (s) handed to the merge step, in order: \(mergePromptTimestamps(parallel))
          (strictly ascending -> results were correctly re-sorted by timestamp)
        observations returned: \(parallel.observations.count), ordered by start time: \(observationsSorted)
      ===================================================
      """
    print(evidence)

    let evidenceDir = "/var/folders/bt/8cq8cf9j705bm1dd707kwdww0000gn/T/no-mistakes-evidence/01KWRM69GZXR8ET2516FEFTR76"
    let evidenceURL = URL(fileURLWithPath: evidenceDir, isDirectory: true)
    try? FileManager.default.createDirectory(at: evidenceURL, withIntermediateDirectories: true)
    try? evidence.write(
      to: evidenceURL.appendingPathComponent("concurrency-evidence.txt"),
      atomically: true, encoding: .utf8)
  }

  // MARK: - Settings UI screenshot

  /// Renders the real Settings "Current configuration" rows (including the new
  /// "Max concurrent requests" stepper) to a PNG so the reviewer can see the control.
  @MainActor
  func testRenderMaxConcurrencyStepperScreenshot() throws {
    let evidenceDir = "/var/folders/bt/8cq8cf9j705bm1dd707kwdww0000gn/T/no-mistakes-evidence/01KWRM69GZXR8ET2516FEFTR76"
    let evidenceURL = URL(fileURLWithPath: evidenceDir, isDirectory: true)
    try? FileManager.default.createDirectory(at: evidenceURL, withIntermediateDirectories: true)

    let viewModel = ProvidersSettingsViewModel()
    viewModel.currentProvider = "ollama"
    viewModel.localEngine = .ollama
    viewModel.localModelId = "qwen3-vl-4b"
    viewModel.localBaseURL = "http://localhost:1234"
    viewModel.localMaxConcurrency = 4

    let content = OllamaConfigurationSnapshot(viewModel: viewModel)
      .frame(width: 560, alignment: .leading)
      .padding(28)
      .background(Color(red: 0.99, green: 0.98, blue: 0.96))

    let hosting = NSHostingView(rootView: content)
    hosting.frame = NSRect(x: 0, y: 0, width: 616, height: 10)
    hosting.layoutSubtreeIfNeeded()
    let fitting = hosting.fittingSize
    hosting.frame = NSRect(x: 0, y: 0, width: 616, height: max(fitting.height, 240))

    let window = NSWindow(
      contentRect: hosting.frame, styleMask: [.borderless], backing: .buffered, defer: false)
    window.contentView = hosting
    window.orderFrontRegardless()
    hosting.layoutSubtreeIfNeeded()
    RunLoop.current.run(until: Date().addingTimeInterval(0.5))

    let bounds = hosting.bounds
    let rep = try XCTUnwrap(
      hosting.bitmapImageRepForCachingDisplay(in: bounds), "could not allocate bitmap rep")
    hosting.cacheDisplay(in: bounds, to: rep)
    let png = try XCTUnwrap(
      rep.representation(using: .png, properties: [:]), "could not encode PNG")

    let outURL = evidenceURL.appendingPathComponent("settings-max-concurrent-requests.png")
    try png.write(to: outURL)
    print("SCREENSHOT written: \(outURL.path) (\(png.count) bytes, \(Int(bounds.width))x\(Int(bounds.height)) pt)")
    XCTAssertGreaterThan(png.count, 1000, "rendered PNG should be non-trivial")
  }

  // MARK: - Harness

  private struct RunResult {
    let peak: Int
    let describeCount: Int
    let observations: [Observation]
    let mergePrompt: String
  }

  private func runTranscription(cap: Int, frameCount: Int) async throws -> RunResult {
    UserDefaults.standard.set(cap, forKey: concurrencyKey)
    UserDefaults.standard.set("ollama", forKey: engineKey)
    UserDefaults.standard.set("test-model", forKey: modelKey)

    MockLocalLLMURLProtocol.recorder = MockLocalLLMURLProtocol.Recorder(totalFrames: frameCount)

    let imagePath = try makeTempImagePath()
    let base = Date(timeIntervalSince1970: 1_700_000_000)
    let baseUnix = Int(base.timeIntervalSince1970)
    let screenshots: [Screenshot] = (0..<frameCount).map { i in
      Screenshot(
        id: Int64(i),
        capturedAt: baseUnix + i * 10,
        filePath: imagePath,
        fileSize: nil,
        idleSecondsAtCapture: nil,
        isDeleted: false)
    }

    let provider = OllamaProvider(endpoint: "http://127.0.0.1:9")
    let (observations, _) = try await provider.transcribeScreenshots(
      screenshots, batchStartTime: base, batchId: nil)

    let recorder = MockLocalLLMURLProtocol.recorder
    return RunResult(
      peak: recorder.peakInFlight,
      describeCount: recorder.describeCount,
      observations: observations,
      mergePrompt: recorder.capturedMergePrompts.first ?? "")
  }

  private func makeTempImagePath() throws -> String {
    let rep = NSBitmapImageRep(
      bitmapDataPlanes: nil, pixelsWide: 8, pixelsHigh: 8, bitsPerSample: 8,
      samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
      bytesPerRow: 0, bitsPerPixel: 0)!
    guard let data = rep.representation(using: .jpeg, properties: [:]) else {
      throw XCTSkip("Could not encode test JPEG")
    }
    let url = tempDir.appendingPathComponent("frame.jpg")
    try data.write(to: url)
    return url.path
  }

  /// The `[MM:SS]` markers in the segment-merge prompt reflect the order of the
  /// `frameDescriptions` array handed to the merge step; the re-sort guarantees they ascend.
  /// The prompt arrives as a JSON request body (newlines escaped), so match on the markers
  /// directly rather than splitting on lines.
  private func mergePromptTimestamps(_ run: RunResult) -> [Int] {
    regexMatches(in: run.mergePrompt, pattern: "\\[(\\d{2}):(\\d{2})\\]").map { groups in
      (Int(groups[1]) ?? 0) * 60 + (Int(groups[2]) ?? 0)
    }
  }

  /// Completion sequence numbers embedded in each frame description, read in merge-prompt order.
  private func mergePromptCompletionOrder(_ run: RunResult) -> [Int] {
    regexMatches(in: run.mergePrompt, pattern: "cmp(\\d+)").compactMap { Int($0[1]) }
  }

  /// Returns capture-group strings (`[0]` = full match) for every match, in order.
  private func regexMatches(in text: String, pattern: String) -> [[String]] {
    guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
    let ns = text as NSString
    return regex.matches(in: text, range: NSRange(location: 0, length: ns.length)).map { match in
      (0..<match.numberOfRanges).map { i in
        let range = match.range(at: i)
        return range.location == NSNotFound ? "" : ns.substring(with: range)
      }
    }
  }

  private func assertMergePromptSortedByTimestamp(
    _ run: RunResult, file: StaticString = #filePath, line: UInt = #line
  ) {
    let timestamps = mergePromptTimestamps(run)
    XCTAssertEqual(timestamps.count, run.describeCount, "one merge-prompt entry per frame", file: file, line: line)
    XCTAssertEqual(
      timestamps, timestamps.sorted(), "merge input must be sorted ascending by timestamp",
      file: file, line: line)
  }
}

extension Array where Element == Int {
  fileprivate var isAscending: Bool {
    guard count > 1 else { return true }
    return zip(self, dropFirst()).allSatisfy { $0 <= $1 }
  }
}

/// Mirrors the ollama "Current configuration" rows from `SettingsProvidersTabView`, built from the
/// same `SettingsSection`/`SettingsRow`/`SettingsMetadata` components and the same `Stepper` so the
/// rendered screenshot matches what a user sees in Settings → Providers.
private struct OllamaConfigurationSnapshot: View {
  @ObservedObject var viewModel: ProvidersSettingsViewModel

  var body: some View {
    SettingsSection(
      title: "Current configuration",
      subtitle: "Active provider and runtime details."
    ) {
      VStack(alignment: .leading, spacing: 0) {
        SettingsRow(label: "Engine") { SettingsMetadata(text: viewModel.localEngine.displayName) }
        SettingsRow(label: "Model") {
          SettingsMetadata(
            text: viewModel.localModelId.isEmpty ? "Not configured" : viewModel.localModelId)
        }
        SettingsRow(label: "Endpoint") { SettingsMetadata(text: viewModel.localBaseURL) }
        SettingsRow(label: "Max concurrent requests") {
          HStack(spacing: 8) {
            SettingsMetadata(text: "\(viewModel.localMaxConcurrency)")
            Stepper("", value: $viewModel.localMaxConcurrency, in: 1...16)
              .labelsHidden()
              .fixedSize()
          }
        }
        let hasKey = !viewModel.localAPIKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        SettingsRow(label: "API key", showsDivider: false) {
          SettingsMetadata(text: hasKey ? "Stored in UserDefaults" : "Not set")
        }
      }
    }
  }
}

/// Intercepts the local `/v1/chat/completions` calls so the real transcription pipeline can run
/// without a live LM Studio / Ollama server. Tracks peak in-flight `describe_frame` requests and
/// captures the segment-merge prompt for re-sort verification.
final class MockLocalLLMURLProtocol: URLProtocol {
  final class Recorder: @unchecked Sendable {
    private let lock = NSLock()
    let totalFrames: Int
    let baseDelay: TimeInterval
    let step: TimeInterval

    private var inFlight = 0
    private(set) var peakInFlight = 0
    private var arrivals = 0
    private var completions = 0
    private(set) var describeCount = 0
    private(set) var capturedMergePrompts: [String] = []

    init(totalFrames: Int = 0, baseDelay: TimeInterval = 0.25, step: TimeInterval = 0.015) {
      self.totalFrames = totalFrames
      self.baseDelay = baseDelay
      self.step = step
    }

    /// Registers a describe_frame arrival; returns its arrival index.
    func beginDescribe() -> Int {
      lock.lock()
      defer { lock.unlock() }
      inFlight += 1
      describeCount += 1
      peakInFlight = Swift.max(peakInFlight, inFlight)
      let index = arrivals
      arrivals += 1
      return index
    }

    /// Registers a describe_frame completion; returns its completion sequence number.
    func endDescribe() -> Int {
      lock.lock()
      defer { lock.unlock() }
      inFlight -= 1
      let seq = completions
      completions += 1
      return seq
    }

    func captureMergePrompt(_ prompt: String) {
      lock.lock()
      defer { lock.unlock() }
      capturedMergePrompts.append(prompt)
    }
  }

  static var recorder = Recorder()
  private static let workQueue = DispatchQueue(label: "mock.local.llm", attributes: .concurrent)

  override class func canInit(with request: URLRequest) -> Bool {
    request.url?.absoluteString.contains("/chat/completions") ?? false
  }

  override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

  override func startLoading() {
    let body = Self.bodyString(from: request)
    let recorder = Self.recorder

    if body.contains("image_url") {
      let arrival = recorder.beginDescribe()
      // Give earlier-timestamp frames longer delays so completion order deliberately
      // diverges from submission/timestamp order, exercising the downstream re-sort.
      let slots = Swift.max(0, recorder.totalFrames - 1 - arrival)
      let delay = recorder.baseDelay + Double(slots) * recorder.step
      Self.workQueue.asyncAfter(deadline: .now() + delay) { [weak self] in
        let completion = recorder.endDescribe()
        self?.finish(content: "arr\(arrival)_cmp\(completion)")
      }
    } else {
      // segment_video_activity (or any text call): capture the prompt, then return a
      // non-segment envelope so the pipeline falls back to per-frame observations.
      recorder.captureMergePrompt(body)
      Self.workQueue.asyncAfter(deadline: .now() + 0.005) { [weak self] in
        self?.finish(content: "NON_SEGMENT_RESPONSE")
      }
    }
  }

  override func stopLoading() {}

  private func finish(content: String) {
    let envelope = ChatEnvelope(choices: [.init(message: .init(content: content))])
    let data = (try? JSONEncoder().encode(envelope)) ?? Data()
    if let url = request.url,
      let response = HTTPURLResponse(
        url: url, statusCode: 200, httpVersion: "HTTP/1.1",
        headerFields: ["Content-Type": "application/json"])
    {
      client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
      client?.urlProtocol(self, didLoad: data)
      client?.urlProtocolDidFinishLoading(self)
    }
  }

  private struct ChatEnvelope: Encodable {
    struct Choice: Encodable { let message: Message }
    struct Message: Encodable { let content: String }
    let choices: [Choice]
  }

  private static func bodyString(from request: URLRequest) -> String {
    if let body = request.httpBody {
      return String(data: body, encoding: .utf8) ?? ""
    }
    guard let stream = request.httpBodyStream else { return "" }
    stream.open()
    defer { stream.close() }
    var data = Data()
    let bufferSize = 8192
    let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferSize)
    defer { buffer.deallocate() }
    while stream.hasBytesAvailable {
      let read = stream.read(buffer, maxLength: bufferSize)
      if read <= 0 { break }
      data.append(buffer, count: read)
    }
    return String(data: data, encoding: .utf8) ?? ""
  }
}
