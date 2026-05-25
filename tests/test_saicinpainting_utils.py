import os

import torch

from saicinpainting.utils import (
    add_prefix_to_keys,
    average_dicts,
    get_has_ddp_rank,
    get_ramp,
    get_shape,
    handle_ddp_parent_process,
    handle_ddp_subprocess,
    LinearRamp,
    LadderRamp,
)


def test_get_shape_tensor():
    tensor = torch.randn(2, 3, 4)
    assert get_shape(tensor) == (2, 3, 4)


def test_get_shape_nested_structures():
    nested = {
        "a": torch.zeros(1, 2),
        "b": [torch.ones(2, 2), {"x": torch.randn(3)}],
    }
    shape = get_shape(nested)
    assert shape["a"] == (1, 2)
    assert shape["b"][0] == (2, 2)
    assert shape["b"][1]["x"] == (3,)


def test_add_prefix_to_keys_and_average_dicts():
    result = add_prefix_to_keys({"loss": 1.0}, "train_")
    assert result == {"train_loss": 1.0}

    average = average_dicts([{"a": 1.0}, {"a": 2.0, "b": 1.0}])
    assert average["a"] > 0 and average["b"] > 0


def test_linear_and_ladder_ramp_behaviors():
    ramp = get_ramp(kind="linear", start_value=0, end_value=1, start_iter=0, end_iter=10)
    assert isinstance(ramp, LinearRamp)
    assert ramp(0) == 0
    assert ramp(10) == 1

    ladder = get_ramp(kind="ladder", start_iters=[5, 10], values=[0, 1, 2])
    assert isinstance(ladder, LadderRamp)
    assert ladder(0) == 0
    assert ladder(7) == 1
    assert ladder(11) == 2


def test_ddp_helpers_no_env_vars():
    for key in ["MASTER_PORT", "NODE_RANK", "LOCAL_RANK", "WORLD_SIZE", "TRAINING_PARENT_WORK_DIR"]:
        os.environ.pop(key, None)

    assert get_has_ddp_rank() is False

    @handle_ddp_subprocess()
    def dummy():
        return "ok"

    assert dummy() is None
