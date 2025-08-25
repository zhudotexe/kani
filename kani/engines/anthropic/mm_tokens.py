# ==== images ====
import math


def tokens_from_image_size(size: tuple[int, int]) -> int:
    """
    Estimate the number of tokens used after providing this image.

    See https://docs.anthropic.com/en/docs/build-with-claude/vision#evaluate-image-size for more details.
    """
    long, short = size
    if long < short:
        long, short = short, long

    # rescale so the larger side is 1568px
    if long > 1568:
        ratio = long / 1568
        long = 1568
        short //= ratio

    return math.ceil((long * short) / 750)
