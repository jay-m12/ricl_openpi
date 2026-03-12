import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_droid_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RiclDroidInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    action_dim: int
    num_retrieved_observations: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def _build_single_observation(self, data: dict, prefix: str) -> dict:
        state = transforms.pad_to_dim(np.asarray(data[f"{prefix}state"]), self.action_dim)

        top_image = _parse_image(data[f"{prefix}top_image"])
        wrist_image = _parse_image(data[f"{prefix}wrist_image"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (top_image, wrist_image, np.zeros_like(top_image))
                image_masks = (np.True_, np.True_, np.False_)

            case _model.ModelType.PI0_FAST:
                right_image = _parse_image(data[f"{prefix}right_image"])
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (top_image, right_image, wrist_image)
                image_masks = (np.True_, np.True_, np.True_)

            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        result = {
            f"{prefix}state": state,
            f"{prefix}image": dict(zip(names, images, strict=True)),
            f"{prefix}image_mask": dict(zip(names, image_masks, strict=True)),
            f"{prefix}prompt": data[f"{prefix}prompt"],
        }

        if f"{prefix}actions" in data:
            result[f"{prefix}actions"] = data[f"{prefix}actions"]

        return result

    def __call__(self, data: dict) -> dict:
        all_prefix = [f"retrieved_{i}_" for i in range(self.num_retrieved_observations)] + ["query_"]

        inputs = {}
        for prefix in all_prefix:
            inputs.update(self._build_single_observation(data, prefix))

        if "exp_lamda_distances" in data:
            inputs["exp_lamda_distances"] = data["exp_lamda_distances"]

        if "inference_time" in data:
            inputs["inference_time"] = data["inference_time"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DroidInputs(transforms.DataTransformFn):
    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        state = np.concatenate([data["observation/joint_position"], data["observation/gripper_position"]])
        state = transforms.pad_to_dim(state, self.action_dim)

        base_image = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DroidOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        return {"actions": np.asarray(data["actions"][:, :8])}
    
@dataclasses.dataclass(frozen=True)
class RiclDroidOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        return {"query_actions": np.asarray(data["query_actions"])}
