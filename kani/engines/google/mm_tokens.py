import math


# ==== images ====
# deprecated, use prompt_len
def tokens_from_image_size(size: tuple[int, int], model_id: str) -> int:
    """
    Estimate the number of tokens used after providing this image.

    See https://ai.google.dev/gemini-api/docs/image-understanding#token_calculation for more details.
    """
    long, short = size  # actually (width, height)
    if long < short:
        long, short = short, long

    if long <= 384 and short <= 384:
        return 258

    n_patches = math.ceil(long / 768) * math.ceil(short / 768)
    return n_patches * 258


# ==== audio ====
# deprecated, use prompt_len
def tokens_from_audio_duration(seconds: float, model_id: str) -> int:
    """
    Estimate the number of tokens used after providing this audio.

    See https://ai.google.dev/gemini-api/docs/audio#count-tokens.
    """
    return math.ceil(seconds * 32)


# ==== video ====
# deprecated, use prompt_len
def tokens_from_video_duration(seconds: float, model_id: str, fps: float = 1, low_res: bool = False) -> int:
    """
    Estimate the number of tokens used after providing this video.

    See https://ai.google.dev/gemini-api/docs/video-understanding#technical-details-video.
    """
    n_frames = math.ceil(seconds * fps)
    tokens_per_frame = 258 if not low_res else 66
    video_tokens = n_frames * tokens_per_frame
    return tokens_from_audio_duration(seconds, model_id) + video_tokens
