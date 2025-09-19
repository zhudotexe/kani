import math
import warnings

from .model_constants import MM_IMAGE_LOW_COST_SCALE, MM_IMAGE_OLD_SCALE


# ==== images ====
def tokens_from_image_size(size: tuple[int, int], model_id: str, low_detail: bool = False) -> int:
    """
    Estimate the number of tokens used after providing this image.

    See https://platform.openai.com/docs/guides/images-vision#calculating-costs for more details.
    """
    if low_detail:
        return 85

    long, short = size  # actually (width, height)
    if any(m in model_id for m in MM_IMAGE_LOW_COST_SCALE):
        return _tokens_from_image_size_low_cost(long, short, model_id)

    # old sizing doesn't care about width vs height, just long vs short
    if long < short:
        long, short = short, long
    return _tokens_from_image_size_old(long, short, model_id)


def _tokens_from_image_size_low_cost(width: int, height: int, model_id: str) -> int:
    n_patches = math.ceil(width / 32) * math.ceil(height / 32)

    if n_patches > 1536:
        # do the scaling documented at
        # https://platform.openai.com/docs/guides/images-vision#gpt-4-1-mini-gpt-4-1-nano-o4-mini
        shrink_factor = math.sqrt(1536 * 32 * 32 / (width * height))
        width = math.ceil(width * shrink_factor)
        height = math.ceil(height * shrink_factor)

        # shrink again to make width a whole number of patches
        shrink_factor = (width // 32) / (width / 32)
        width = math.floor(width * shrink_factor)
        height = math.floor(height * shrink_factor)

        n_patches = math.ceil(width / 32) * math.ceil(height / 32)
        print(n_patches)

    # get multiplier per model
    multiplier = MM_IMAGE_LOW_COST_SCALE.get(model_id, 1)
    return math.ceil(n_patches * multiplier)


def _tokens_from_image_size_old(long: int, short: int, model_id: str) -> int:
    # rescale so the larger side is 2048px
    if long > 2048:
        ratio = long / 2048
        long = 2048
        short //= ratio

    # rescale so the smaller side is 768px
    if short > 768:
        ratio = short / 768
        short = 768
        long //= ratio

    n_patches = math.ceil(long / 512) * math.ceil(short / 512)

    # get base + scale per model
    if model_id not in MM_IMAGE_OLD_SCALE:
        base, scale = MM_IMAGE_OLD_SCALE[None]
        warnings.warn(
            f"The multimodal image scale was not found for the model {model_id!r}. Defaulting to {scale} tokens per"
            f" 512x512 image patch (after scaling image to {long}x{short}, ={n_patches} patches) plus {base} base"
            " tokens."
        )
    else:
        base, scale = MM_IMAGE_OLD_SCALE[model_id]
    return base + (n_patches * scale)


# ==== audio ====
def tokens_from_audio_duration(seconds: float, model_id: str) -> int:
    """
    Estimate the number of tokens used after providing this audio.
    """
    # based on experimentation, doesn't seem to be documented anywhere
    return math.ceil(seconds * 10)
