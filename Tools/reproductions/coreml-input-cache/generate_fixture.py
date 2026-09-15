"""Run with: uv run --python 3.12 --with coremltools==9.0 generate_fixture.py"""
from pathlib import Path

from coremltools.models import datatypes
from coremltools.models.neural_network import NeuralNetworkBuilder
from coremltools.models.utils import save_spec

builder = NeuralNetworkBuilder(
    [("x", datatypes.Array(1))],
    [("y", datatypes.Array(1))],
    use_float_arraytype=True,
)
builder.add_activation(
    name="identity", non_linearity="LINEAR", input_name="x", output_name="y", params=[1.0, 0.0]
)
save_spec(builder.spec, str(Path(__file__).parent / "Resources" / "identity.mlmodel"))
