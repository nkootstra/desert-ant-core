import CoreML
import Foundation
import Inference

@main
struct Reproduce {
    static func main() async {
        exit(await check())
    }

    private static func check() async -> Int32 {
        do {
            let mode = CommandLine.arguments.dropFirst().first ?? "stale-provider"
            guard ["baseline", "stale-provider", "type-change"].contains(mode) else {
                print("Use baseline, stale-provider, or type-change.")
                return 2
            }
            guard let source = Bundle.module.url(forResource: "identity", withExtension: "mlmodel") else {
                print("SETUP FAILED: the bundled identity model is missing.")
                return 2
            }
            let compiled = try await MLModel.compileModel(at: source)
            defer { try? FileManager.default.removeItem(at: compiled) }
            let session = try inferenceSession(modelPath: compiled.path, computeUnits: .cpuOnly)
            let input = Tensor(float32: [7], shape: [1])
            let warm = try await session.run(inputs: ["x": input], outputs: ["y"])
            guard warm.first?.float32Values == [7] else {
                print("SETUP FAILED: identity model did not return [7].")
                return 2
            }
            print("CONTROL: valid input x=[7] returned y=[7].")

            if mode == "baseline" {
                let changed = try await session.run(
                    inputs: ["x": Tensor(float32: [9], shape: [1])], outputs: ["y"]
                )
                guard changed.first?.float32Values == [9] else {
                    print("SETUP FAILED: a valid cache-hit call did not update the input.")
                    return 2
                }
                print("CONTROL: a valid cache-hit call returned the new value [9].")
                let fresh = try inferenceSession(modelPath: compiled.path, computeUnits: .cpuOnly)
                do {
                    _ = try await fresh.run(inputs: [:], outputs: ["y"])
                    print("SETUP FAILED: a fresh session accepted the missing required input.")
                    return 2
                } catch {
                    print("CONTROL: a fresh session rejected the missing required input: \(error)")
                }
                let rejected = try await rejectsInt64(session, shape: [2], values: [7, 8])
                guard rejected else { return 2 }
                print("PASS: fixture and int64 rejection controls passed.")
                return 0
            }

            if mode == "stale-provider" {
                let rejected = try await rejectsInt64(session, shape: [2], values: [7, 8])
                guard rejected else { return 2 }
                do {
                    let result = try await session.run(inputs: [:], outputs: ["y"])
                    print("FAIL: missing required input was accepted after a rejected cache rebuild.")
                    print("OBSERVED: y=\(result.first?.float32Values ?? []). Expected an error.")
                    return 1
                } catch {
                    print("PASS: missing required input was rejected after the failed rebuild: \(error)")
                }
                return 0
            }

            // This case can copy eight source bytes into a cached float32 element.
            // Run it as a separate process, with no application data loaded.
            print("PROBE: replacing float32 with int64 at the same name and shape.")
            fflush(stdout)
            let bits = Int64(Float(7).bitPattern)
            if try await rejectsInt64(session, shape: [1], values: [bits]) {
                print("PASS: int64 was rejected on the cache-hit path.")
            } else {
                print("FAIL: int64 bypassed validation on the cache-hit path.")
                return 1
            }
            return 0
        } catch {
            print("SETUP FAILED: \(error)")
            return 2
        }
    }

    private static func rejectsInt64(
        _ session: any InferenceSession, shape: [Int], values: [Int64]
    ) async throws -> Bool {
        do {
            let result = try await session.run(
                inputs: ["x": Tensor(int64: values, shape: shape)], outputs: ["y"]
            )
            print("OBSERVED: int64 input returned y=\(result.first?.float32Values ?? []).")
            return false
        } catch InferenceError.invalidTensor(let message) {
            print("CONTROL: int64 was rejected: \(message)")
            return true
        }
    }
}
