import base64
from pathlib import Path
from typing import IO

from pydantic import model_serializer, model_validator

from kani import MessagePart
from kani.utils.typing import PathLike


# ===== PDF Documents =====
class AnthropicPDFFilePart(MessagePart):
    """
    A part representing a PDF, used for Anthropic's PDF file input.
    """

    # TODO: CLI register @handler
    # todo: maybe refactor into a generic binaryfilepart?

    data: bytes

    # ==== constructors ====
    @classmethod
    def from_file(cls, fp: PathLike | IO):
        """Create an AnthropicPDFFilePart from a local PDF file."""
        return cls(data=Path(fp).read_bytes())

    @classmethod
    def from_bytes(cls, data: bytes):
        """Create an AnthropicPDFFilePart from raw binary data."""
        return cls(data=data)

    @classmethod
    def from_b64(cls, data: str):
        """Create an AnthropicPDFFilePart from Base64-encoded binary data."""
        return cls.from_bytes(base64.b64decode(data))

    # ==== representations ====
    def as_bytes(self) -> bytes:
        """Return the raw data."""
        return self.data

    def as_b64(self) -> str:
        """Return the binary data encoded in a base64 string."""
        return base64.b64encode(self.as_bytes()).decode()

    # ==== helpers ====
    @property
    def size(self) -> int:
        """The size of the file, in bytes."""
        return len(self.data)

    # ==== serdes ====
    @model_serializer(when_used="json")
    def _serialize_anthropic_pdf_file_part(self) -> dict[str, str]:
        """When we serialize to JSON, save the data as B64"""
        return {"data": self.as_b64()}

    # noinspection PyNestedDecorators
    @model_validator(mode="wrap")
    @classmethod
    def _validate_anthropic_pdf_file_part(cls, v, nxt):
        """If the value is the B64 we saved, try loading it that way"""
        if isinstance(v, dict) and "data" in v:
            return cls.from_b64(v["data"])
        return nxt(v)


# ===== Generic =====
class AnthropicUnknownPart(MessagePart):
    """
    A generic unknown response part from the server.

    This generally corresponds to an Anthropic-specific feature. The raw response data is accessible in :attr:`data`,
    and will be sent back to the language model in future rounds correctly. Will not be sent to other engines.
    """

    type: str
    data: dict
