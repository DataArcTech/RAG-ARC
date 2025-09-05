from __future__ import annotations

from typing import get_args, get_origin, Literal, TYPE_CHECKING
from pydantic import BaseModel, Field, field_validator
from typing import Annotated

if TYPE_CHECKING:
    from config.module import AbstractModule


class AbstractConfig(BaseModel):
    """
    Enforce that *each direct subclass* declares:
        type: Literal["<TAG>"] = "<TAG>"
    """

    def build(self) -> "AbstractModule":
        """
        This method is used to generate the class corresponding to this config.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement build() method")

    # Runs whenever a subclass is created
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls is AbstractConfig:
            return  # don't check the abstract base itself

        ann = cls.__dict__.get("__annotations__", {})

        # 1) must declare 'type' in THIS class (not just inherit)
        if "type" not in ann:
            raise TypeError(
                f"{cls.__name__} must declare `type: Literal['TAG'] = 'TAG'`"
            )

        # 2) its annotation must be Literal["..."]
        typ_ann = ann["type"]
        # Handle string annotations (from __future__ import annotations)
        if isinstance(typ_ann, str):
            # For string annotations, we can't easily check the origin
            # Just check that it looks like a Literal
            if not typ_ann.startswith("Literal["):
                raise TypeError(
                    f"{cls.__name__}.type must be annotated as Literal['TAG']"
                )
        else:
            if get_origin(typ_ann) is not Literal:
                raise TypeError(
                    f"{cls.__name__}.type must be annotated as Literal['TAG']"
                )

        # 3) default value must equal the literal
        default = cls.__dict__.get("type", None)
        if isinstance(typ_ann, str):
            # For string annotations, we can't easily extract the literal value
            # Just check that we have a default value
            if default is None:
                raise TypeError(f"{cls.__name__}.type must have a default value")
        else:
            lit_args = get_args(typ_ann)
            if len(lit_args) != 1 or not isinstance(lit_args[0], str):
                raise TypeError(
                    f"{cls.__name__}.type must be Literal['<single string>']"
                )
            if default != lit_args[0]:
                raise TypeError(
                    f"{cls.__name__}.type default must equal {lit_args[0]!r}"
                )

    # Runtime guard: parsed JSON cannot lie about the tag
    @field_validator("type", check_fields=False)
    @classmethod
    def _validate_type_literal(cls, v: str) -> str:
        # Only validate if this class has a type field
        if "type" not in cls.__annotations__:
            return v

        # Get the default value which should match the literal
        default = cls.__dict__.get("type", None)
        if default is None:
            return v

        # Validate against the default value
        if v != default:
            raise ValueError(f"type must be {default!r}")
        return v
