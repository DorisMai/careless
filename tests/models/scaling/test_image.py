from careless.models.scaling.image import ImageScaler
import pytest
import tensorflow as tf
import numpy as np

from careless.utils.device import disable_gpu
status = disable_gpu()
assert status


def ImageScaler_test(mc_samples, inputs):
    image_scaler = ImageScaler(10) #<-- 10 is much bigger than the actual value
    q = image_scaler(inputs)
    assert image_scaler.get_metadata(inputs).shape[0] == q.shape[-2]
    assert q.shape[-1] == 1


@pytest.mark.parametrize("mc_samples", [1, 3])
def test_ImageScaler_laue(mc_samples, laue_inputs):
    ImageScaler_test(mc_samples, laue_inputs)

@pytest.mark.parametrize("mc_samples", [1, 3])
def test_ImageScaler_mono(mc_samples, mono_inputs):
    ImageScaler_test(mc_samples, mono_inputs)
