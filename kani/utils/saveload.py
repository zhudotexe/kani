import dataclasses
import hashlib
import zipfile
from typing import Any

from kani.models import BaseModel, ChatMessage
from kani.utils.typing import PathLike


# ==== main ====
class SavedKani(BaseModel):
    version: int = 1
    always_included_messages: list[ChatMessage]
    chat_history: list[ChatMessage]

    def model_dump(self, **kwargs) -> dict[str, Any]:
        return super().model_dump(serialize_as_any=True, **kwargs)

    def model_dump_json(self, **kwargs) -> str:
        return super().model_dump_json(serialize_as_any=True, **kwargs)


def save(fp: PathLike, inst, *, save_format: str, **kwargs):
    # create a Pydantic model for the saved attrs
    data = SavedKani(always_included_messages=inst.always_included_messages, chat_history=inst.chat_history)

    if save_format == "kani":
        # zip w/ manifest file for multimodal attachments
        with zipfile.ZipFile(fp, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            ctx = KaniZipSaveContext(zf=zf)
            # the model_dump_json should write to zf when SAVELOAD_CONTEXT_KEY is provided
            index = data.model_dump_json(context={SAVELOAD_CONTEXT_KEY: ctx}, **kwargs)
            with zf.open("index.json", mode="w") as f:
                f.write(index.encode("utf-8"))

    elif save_format == "json":
        # save using legacy JSON
        with open(fp, "w", encoding="utf-8") as f:
            f.write(data.model_dump_json(**kwargs))
    else:
        raise ValueError("save_format must be either 'kani' or 'json'.")


def load(fp: PathLike, **kwargs) -> SavedKani:
    # test file format
    if zipfile.is_zipfile(fp):
        # zipfile
        with zipfile.ZipFile(fp, mode="r") as zf:
            ctx = KaniZipSaveContext(zf=zf)
            with zf.open("index.json") as f:
                data = f.read().decode(encoding="utf-8")
            return SavedKani.model_validate_json(data, context={SAVELOAD_CONTEXT_KEY: ctx}, **kwargs)
    else:
        # json
        with open(fp, encoding="utf-8") as f:
            data = f.read()
        return SavedKani.model_validate_json(data, **kwargs)


# ==== zip mode ====
# to utilize multi-file saving, a model that is being saved (usually a MessagePart) should use a wrap-mode model
# serializer and check for this key in the serializationinfo.context object
SAVELOAD_CONTEXT_KEY = "kani.saveload.context"


@dataclasses.dataclass
class KaniZipSaveContext:
    zf: zipfile.ZipFile

    def save_bytes(self, data: bytes, suffix: str = "") -> str:
        """
        Save the given bytes to the zip file and return its path.
        Filename is automatically determined by SHA256 hash.
        If *suffix* is given, the filename will end with the given suffix.
        """
        the_hash = hashlib.sha256(data)
        digest = the_hash.hexdigest()
        fp = f"blobs/{digest[:2]}/{digest}{suffix}"
        with self.zf.open(fp, mode="w") as f:
            f.write(data)
        return fp

    def load_bytes(self, fp: str) -> bytes:
        """
        Read the bytes from the given path in the archive.
        """
        with self.zf.open(fp, mode="r") as f:
            return f.read()


def get_ctx(info) -> KaniZipSaveContext | None:
    """Get the KaniZipSaveContext from a SerializationInfo/ValidationInfo object."""
    if info.context and SAVELOAD_CONTEXT_KEY in info.context:
        ctx = info.context[SAVELOAD_CONTEXT_KEY]
        assert isinstance(ctx, KaniZipSaveContext)
        return ctx
    return None
