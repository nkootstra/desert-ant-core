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

    @Test func concurrentRunsKeepTheirOwnInputValues() async throws {
        try await withSession { session in
            try await withThrowingTaskGroup(of: Void.self) { group in
                for value in 1...16 {
                    group.addTask {
                        let tensor = value.isMultiple(of: 2)
                            ? Tensor(int32: [Int32(value)], shape: [1])
                            : Tensor(float32: [Float(value)], shape: [1])
                        let result = try await session.run(inputs: ["x": tensor], outputs: ["y"])
                        #expect(result.first?.float32Values == [Float(value)])
                    }
                }
                try await group.waitForAll()
            }
        }
    }

    @Test func validShapeChangesUseCurrentInputValues() async throws {
        try await withSession(resource: "identity-flexible") { session in
            _ = try await session.run(
                inputs: ["x": Tensor(float32: [7], shape: [1])], outputs: ["y"]
            )
            let wider = try await session.run(
                inputs: ["x": Tensor(float32: [9, 11], shape: [2])], outputs: ["y"]
            )
            #expect(wider.first?.count == 2)
            #expect(wider.first?.float32Values == [9, 11])
            let narrow = try await session.run(
                inputs: ["x": Tensor(float32: [13], shape: [1])], outputs: ["y"]
            )
            #expect(narrow.first?.float32Values == [13])
        }
    }

    @Test func float32InputsRemainCorrectForFloat16ModelsAfterRejection() async throws {
        let source = try #require(Bundle.module.url(forResource: "plus-one-half", withExtension: "mlpackage"))
        let compiled = try await MLModel.compileModel(at: source)
        defer { try? FileManager.default.removeItem(at: compiled) }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuOnly
        let model = try MLModel(contentsOf: compiled, configuration: configuration)
        try #require(model.modelDescription.inputDescriptionsByName["x"]?.multiArrayConstraint?.dataType == .float16)
        try #require(model.modelDescription.outputDescriptionsByName["y"]?.multiArrayConstraint?.dataType == .float16)
        let session = try inferenceSession(modelPath: compiled.path, computeUnits: .cpuOnly)
        let first = try await session.run(
            inputs: ["x": Tensor(float32: [7.25, -2.5], shape: [1, 2])], outputs: ["y"]
        )
        #expect(first.first?.float32Values == [8.25, -1.5])
        do {
            _ = try await session.run(
                inputs: ["x": Tensor(int64: [7, 9], shape: [1, 2])], outputs: ["y"]
            )
            Issue.record("CoreML must reject int64 input for a float16 model.")
        } catch InferenceError.invalidTensor {}
        let recovered = try await session.run(
            inputs: ["x": Tensor(float32: [0.5, 10], shape: [1, 2])], outputs: ["y"]
        )
        #expect(recovered.first?.float32Values == [1.5, 11])
    }

    private func withSession(
        resource: String = "identity", fileExtension: String = "mlmodel",
        _ check: (any InferenceSession) async throws -> Void
    ) async throws {
        let source = try #require(Bundle.module.url(forResource: resource, withExtension: fileExtension))
        let compiled = try await MLModel.compileModel(at: source)
        defer { try? FileManager.default.removeItem(at: compiled) }
        let session = try inferenceSession(modelPath: compiled.path, computeUnits: .cpuOnly)
        try await check(session)
    }
}
#endif
