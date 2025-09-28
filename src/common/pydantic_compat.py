"""Compatibility shim between Pydantic v1 and v2 APIs."""

from __future__ import annotations

from pydantic import BaseModel


class CompatBaseModel(BaseModel):
    """BaseModel with v2-style helpers available on v1 installations."""

    if not hasattr(BaseModel, "model_dump"):
        def model_dump(self, *args, **kwargs):  # type: ignore[override]
            kwargs = dict(kwargs)
            kwargs.pop("mode", None)
            return self.dict(*args, **kwargs)

    if not hasattr(BaseModel, "model_dump_json"):
        def model_dump_json(self, *args, **kwargs):  # type: ignore[override]
            kwargs = dict(kwargs)
            kwargs.pop("mode", None)
            return self.json(*args, **kwargs)

    if not hasattr(BaseModel, "model_copy"):
        def model_copy(self, *args, **kwargs):  # type: ignore[override]
            return self.copy(*args, **kwargs)

    if not hasattr(BaseModel, "model_validate"):
        @classmethod
        def model_validate(cls, obj, *args, **kwargs):  # type: ignore[override]
            return cls.parse_obj(obj, *args, **kwargs)

    if not hasattr(BaseModel, "model_construct"):
        @classmethod
        def model_construct(cls, *args, **kwargs):  # type: ignore[override]
            return cls.construct(*args, **kwargs)
