#if canImport(CoreML)
import CoreML
import Foundation
import Inference
import Testing

struct CoreMLSessionTests {
    @Test func int64InputIsRejectedAfterValidRun() async throws {
        try await withSession { session in
            _ = try await session.run(
                inputs: ["x": Tensor(float32: [7], shape: [1])], outputs: ["y"]
            )
            do {
                _ = try await session.run(
                    inputs: ["x": Tensor(int64: [Int64(Float(7).bitPattern)], shape: [1])], outputs: ["y"]
                )
                Issue.record("CoreML must reject int64 input after a valid run.")
            } catch InferenceError.invalidTensor {}
        }
    }

    @Test func missingInputIsRejectedAfterFailedRebuild() async throws {
        try await withSession { session in
            _ = try await session.run(
                inputs: ["x": Tensor(float32: [7], shape: [1])], outputs: ["y"]
            )
            await #expect(throws: InferenceError.self) {
                try await session.run(
                    inputs: ["x": Tensor(int64: [7, 8], shape: [2])], outputs: ["y"]
                )
            }
            await #expect(throws: (any Error).self) {
                try await session.run(inputs: [:], outputs: ["y"])
            }
        }
    }

    @Test func supportedTypeChangesUseCurrentInputValues() async throws {
        try await withSession { session in
            _ = try await session.run(
                inputs: ["x": Tensor(float32: [7], shape: [1])], outputs: ["y"]
            )
            let integer = try await session.run(
                inputs: ["x": Tensor(int32: [9], shape: [1])], outputs: ["y"]
            )
            #expect(integer.first?.float32Values == [9])
            let floating = try await session.run(
                inputs: ["x": Tensor(float32: [11.5], shape: [1])], outputs: ["y"]
            )
            #expect(floating.first?.float32Values == [11.5])
        }
    }

    @Test func missingInputIsRejectedAfterInvalidShape() async throws {
        try await withSession { session in
            _ = try await session.run(
                inputs: ["x": Tensor(float32: [7], shape: [1])], outputs: ["y"]
            )
            // Use a supported element type to reach native array construction.
            await #expect(throws: (any Error).self) {
                try await session.run(
                    inputs: ["x": Tensor(float32: [7], shape: [-1, -1])], outputs: ["y"]
                )
            }
            await #expect(throws: (any Error).self) {
                try await session.run(inputs: [:], outputs: ["y"])
            }
        }
    }

    @Test func repeatedRunsUseCurrentInputValues() async throws {
        try await withSession { session in
            let first = try await session.run(
                inputs: ["x": Tensor(float32: [7], shape: [1])], outputs: ["y"]
            )
            #expect(first.first?.float32Values == [7])
            let second = try await session.run(
                inputs: ["x": Tensor(float32: [9], shape: [1])], outputs: ["y"]
            )
            #expect(second.first?.float32Values == [9])
        }
    }

    @Test func validInputRecoversAfterRejectedInputs() async throws {
        try await withSession { session in
            _ = try await session.run(
                inputs: ["x": Tensor(float32: [7], shape: [1])], outputs: ["y"]
            )
            await #expect(throws: InferenceError.self) {
                try await session.run(
                    inputs: ["x": Tensor(int64: [7], shape: [1])], outputs: ["y"]
                )
            }
            await #expect(throws: (any Error).self) {
                try await session.run(
                    inputs: ["x": Tensor(float32: [7], shape: [-1, -1])], outputs: ["y"]
                )
            }
            let result = try await session.run(
                inputs: ["x": Tensor(float32: [9], shape: [1])], outputs: ["y"]
            )
            #expect(result.first?.float32Values == [9])
        }
    }

    private func withSession(
        _ check: (any InferenceSession) async throws -> Void
    ) async throws {
        let source = try #require(Bundle.module.url(forResource: "identity", withExtension: "mlmodel"))
        let compiled = try await MLModel.compileModel(at: source)
        defer { try? FileManager.default.removeItem(at: compiled) }
        let session = try inferenceSession(modelPath: compiled.path, computeUnits: .cpuOnly)
        try await check(session)
    }
}
#endif
