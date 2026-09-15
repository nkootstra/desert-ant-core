# CoreML input cache reproduction (S16-F1)

This is an opt-in reproduction. It uses the public `inferenceSession` factory and `InferenceSession.run`. It does not change the backend. It does not download model weights. The bundled identity model has one required float32 input `x` and one output `y`, both with shape `[1]`.

Run from the repository root on macOS with Swift 6.2 or later:

```sh
DAL_USAGE_DISABLED=1 DAL_COREML_COMPUTE_UNITS=cpu swift run --package-path Tools/reproductions/coreml-input-cache Reproduce baseline
DAL_USAGE_DISABLED=1 DAL_COREML_COMPUTE_UNITS=cpu swift run --package-path Tools/reproductions/coreml-input-cache Reproduce stale-provider
```

The baseline must return exit code 0. The stale-provider check returns exit code 1 when the bug is present. It must return exit code 0 after a fix. Exit code 2 means that setup or a control failed.

The stale-provider check warms the session with `x=[7]`. It then sends an int64 tensor with shape `[2]`. This must fail before the copy. It then omits `x`. The third call must fail. A result of `[7]` proves that the session reused the old provider.

The separate type-change check can access memory beyond the cached element. Run it only in its own process:

```sh
DAL_USAGE_DISABLED=1 DAL_COREML_COMPUTE_UNITS=cpu swift run --package-path Tools/reproductions/coreml-input-cache Reproduce type-change
```

It warms the session with float32, then sends int64 with the same name and shape. The call must throw `InferenceError.invalidTensor`. Exit code 1 proves that validation was bypassed. A signal is a separate memory failure. A returned value does not prove memory safety.

To regenerate the fixture, run from this directory:

```sh
uv run --python 3.12 --with coremltools==9.0 generate_fixture.py
```

The fixture uses Apple's [NeuralNetworkBuilder](https://apple.github.io/coremltools/source/coremltools.models.neural_network.html). The reproduction compiles it on the host with [MLModel.compileModel](https://developer.apple.com/documentation/coreml/mlmodel/compilemodel(at:)-45ao6).

Before the fix, verified at `ca4a167` on macOS 26.5 (25F71), arm64, with Apple Swift 6.3.3:

| Check | Exit code | Observed result |
| --- | --- | --- |
| baseline | 0 | Valid cache hits update values; a fresh session rejects missing input. |
| stale-provider | 1 | Missing input returned the old `[7]` after the failed rebuild. |
| type-change | 1 | Same-shape int64 was accepted and returned `[7]`. |

The first failure comes from clearing the cached arrays before a throwing rebuild while retaining the old provider. Empty inputs then match the empty array cache. The second failure comes from matching only input names and shapes, which skips element validation on cache hits. No crash or physical memory overwrite was observed.

These are checks for advanced callers that change inputs on a reused public session. Normal model SDK inputs keep fixed element types. The initial reproduction commit (`ca4a167`) left the backend unchanged.

After the fix, all three modes return exit code 0. The cache owns its arrays and provider as one complete value. A failed construction keeps the previous value complete. Cache matching validates the native element type before any input copy. Supported type changes rebuild the arrays.

The retained public-interface tests are in `Tests/InferenceTests/CoreMLSessionTests.swift`. They cover both reproduced failures, a native float32 array construction failure, valid cache hits, supported type changes, and recovery. Run them from the repository root:

```sh
DAL_USAGE_DISABLED=1 DAL_COREML_COMPUTE_UNITS=cpu xcrun swift test --scratch-path .build-host -c release --disable-xctest --filter CoreMLSessionTests
```

The test resource is an identical copy of this reproduction's 67-byte identity model. These CPU fixture tests do not establish shipping-model, GPU, or ANE behavior.

The validation fixtures also cover valid input shape changes and native float16 input/output. The generator writes these fixtures into `Tests/InferenceTests/Resources`. The flexible neural-network fixture can return a five-dimensional native array; its test checks element count and values. The float16 test checks the compiled model's declared types before inference. A concurrent test checks that one shared session returns each caller's own values while supported input types change.

Fixture APIs: Apple's [flexible input shapes](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html) and [MIL Builder](https://apple.github.io/coremltools/docs/source/coremltools.converters.mil.html).
