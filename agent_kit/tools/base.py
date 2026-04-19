"""Tool abstract base + decorator for wrapping async functions as tools.

Two ways to declare a tool:

1. Subclass ``Tool``, declare a pydantic args model, implement ``run``.
2. Decorate an async function with ``@tool`` — the args model is derived
   from the function signature via pydantic ``create_model``.

Both styles produce an object that Agent Kit can register and expose to any
supported provider. The JSON schema passed to the LLM is generated directly
from the pydantic args model so there's exactly one source of truth.

Example:
    >>> from pydantic import BaseModel
    >>> from agent_kit.tools.base import Tool
    >>>
    >>> class AddArgs(BaseModel):
    ...     a: int
    ...     b: int
    ...
    >>> class AddTool(Tool[AddArgs]):
    ...     name = "add"
    ...     description = "Add two integers."
    ...     args_model = AddArgs
    ...     async def run(self, args: AddArgs) -> int:
    ...         return args.a + args.b
"""

from __future__ import annotations

import asyncio
import inspect
import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar, Generic, TypeVar, get_type_hints

from pydantic import BaseModel, ConfigDict, create_model

ArgsT = TypeVar("ArgsT", bound=BaseModel)


class Tool(ABC, Generic[ArgsT]):
    """Abstract tool: pydantic args in, JSON-serialisable result out.

    Subclasses set the class attributes ``name``, ``description``, and
    ``args_model`` and implement the async ``run`` method. Instances may
    also shadow these with instance attributes when the tool is built
    dynamically (e.g. by the ``@tool`` decorator).
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    args_model: ClassVar[type[BaseModel]]

    @abstractmethod
    async def run(self, args: ArgsT) -> Any:
        """Execute the tool with validated arguments and return a result."""

    def json_schema(self) -> dict[str, Any]:
        """Return a JSON-schema dict describing the tool's arguments.

        Example:
            >>> class Args(BaseModel):
            ...     x: int
            >>> class T(Tool[Args]):
            ...     name = "t"
            ...     description = "noop"
            ...     args_model = Args
            ...     async def run(self, args: Args) -> int:
            ...         return args.x
            >>> "properties" in T().json_schema()
            True
        """
        schema = self.args_model.model_json_schema()
        return _inline_refs(schema)

    def to_spec(self) -> dict[str, Any]:
        """Return a provider-agnostic tool spec."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.json_schema(),
        }

    async def invoke(self, raw_args: dict[str, Any]) -> str:
        """Validate args, execute, and stringify the tool result."""
        args = self.args_model.model_validate(raw_args)
        result = await self.run(args)  # type: ignore[arg-type]
        return _stringify(result)


def _stringify(result: Any) -> str:
    """Coerce an arbitrary tool result to a string safely."""
    if isinstance(result, str):
        return result
    if isinstance(result, BaseModel):
        return result.model_dump_json()
    try:
        return json.dumps(result, default=str)
    except TypeError:
        return str(result)


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline ``$ref``s in a schema; many provider APIs reject them."""
    if not isinstance(schema, dict):
        return schema
    defs = schema.pop("$defs", None) or schema.pop("definitions", None) or {}

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node and len(node) == 1:
                key = node["$ref"].rsplit("/", 1)[-1]
                return walk(dict(defs.get(key, {})))
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(i) for i in node]
        return node

    result = walk(schema)
    assert isinstance(result, dict)
    return result


# ---------- @tool decorator ----------


def tool(
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[..., Awaitable[Any]]], Tool[BaseModel]]:
    """Decorator that converts an async function into a ``Tool`` instance.

    The function must be ``async`` and annotate its parameters. Parameter
    defaults become the tool's defaults. The docstring becomes the tool
    description unless ``description`` is supplied.

    Example:
        >>> from agent_kit.tools.base import tool
        >>> @tool()
        ... async def echo(text: str) -> str:
        ...     '''Echo a string back.'''
        ...     return text
        >>> echo.name
        'echo'
    """

    def decorator(fn: Callable[..., Awaitable[Any]]) -> Tool[BaseModel]:
        if not asyncio.iscoroutinefunction(fn):
            raise TypeError(f"@tool expects an async function, got {fn!r}")
        tool_name = name or fn.__name__
        tool_desc = description or (fn.__doc__ or "").strip() or tool_name
        args_model = _model_from_signature(fn, tool_name)
        return _FunctionTool(fn, tool_name, tool_desc, args_model)

    return decorator


def _model_from_signature(fn: Callable[..., Any], name: str) -> type[BaseModel]:
    """Build a pydantic model from an async function's signature."""
    sig = inspect.signature(fn)
    hints = get_type_hints(fn)
    fields: dict[str, Any] = {}
    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError(f"tool {name!r} cannot use *args / **kwargs")
        annotation = hints.get(pname, Any)
        default = param.default if param.default is not inspect.Parameter.empty else ...
        fields[pname] = (annotation, default)
    model_name = f"{name.title().replace('_', '')}Args"
    return create_model(
        model_name,
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


class _FunctionTool(Tool[BaseModel]):
    """Concrete Tool produced by the ``@tool`` decorator."""

    name: ClassVar[str] = "_function_tool"
    description: ClassVar[str] = ""

    def __init__(
        self,
        fn: Callable[..., Awaitable[Any]],
        name: str,
        description: str,
        args_model: type[BaseModel],
    ) -> None:
        """Bind the target function and metadata to this tool instance."""
        self._fn = fn
        # Instance attributes shadow the ClassVars so each decorated tool
        # carries its own name / description / args_model.
        self.name = name  # type: ignore[misc]
        self.description = description  # type: ignore[misc]
        self.args_model = args_model  # type: ignore[misc]

    async def run(self, args: BaseModel) -> Any:
        """Invoke the wrapped function with validated arguments."""
        return await self._fn(**args.model_dump())


__all__ = ["Tool", "tool"]
