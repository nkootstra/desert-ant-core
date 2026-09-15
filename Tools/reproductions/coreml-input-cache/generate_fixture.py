"""Run with: uv run --python 3.12 --with coremltools==9.0 generate_fixture.py"""
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.models import datatypes
from coremltools.models.neural_network import NeuralNetworkBuilder
from coremltools.models.utils import save_spec

here = Path(__file__).parent
resources = here.parents[2] / "Tests" / "InferenceTests" / "Resources"
builder = NeuralNetworkBuilder(
    [("x", datatypes.Array(1))],
    [("y", datatypes.Array(1))],
    use_float_arraytype=True,
)
builder.add_activation(
    name="identity", non_linearity="LINEAR", input_name="x", output_name="y", params=[1.0, 0.0]
)
save_spec(builder.spec, str(here / "Resources" / "identity.mlmodel"))
save_spec(builder.spec, str(resources / "identity.mlmodel"))
shape_range = builder.spec.description.input[0].type.multiArrayType.shapeRange.sizeRanges.add()
shape_range.lowerBound = 1
shape_range.upperBound = 4
save_spec(builder.spec, str(resources / "identity-flexible.mlmodel"))


@mb.program(input_specs=[mb.TensorSpec(shape=(1, 2), dtype=types.fp16)], opset_version=ct.target.iOS16)
def plus_one(x):
    return mb.add(x=x, y=np.float16(1), name="y")


model = ct.convert(
    plus_one,
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS16,
    compute_precision=ct.precision.FLOAT16,
)
model.save(str(resources / "plus-one-half.mlpackage"))

# Keep the empty weights directory referenced by the package manifest in git.
(resources / "plus-one-half.mlpackage" / "Data" / "com.apple.CoreML" / "weights" / ".keep").touch()
