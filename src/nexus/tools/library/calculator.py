"""Safe arithmetic calculator tool using AST walking — no eval()."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from nexus.tools.library._base import err

_SAFE_OPERATORS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_SAFE_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}

_SAFE_NAMES: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}


def _eval_node(node: ast.expr) -> int | float:
    """Evaluate one AST node, preserving int vs float so stdlib fns get correct types."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in _SAFE_NAMES:
            raise ValueError(f"Unknown name: {node.id!r}")
        return _SAFE_NAMES[node.id]
    if isinstance(node, ast.BinOp):
        bin_op_type: type[Any] = type(node.op)
        if bin_op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {bin_op_type.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        result: int | float = _SAFE_OPERATORS[bin_op_type](left, right)
        return result
    if isinstance(node, ast.UnaryOp):
        unary_op_type: type[Any] = type(node.op)
        if unary_op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {unary_op_type.__name__}")
        operand = _eval_node(node.operand)
        unary_result: int | float = _SAFE_OPERATORS[unary_op_type](operand)
        return unary_result
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCTIONS:
            raise ValueError(f"Unknown or disallowed function: {getattr(node.func, 'id', '?')!r}")
        fn = _SAFE_FUNCTIONS[node.func.id]
        args = [_eval_node(a) for a in node.args]
        call_result: int | float = fn(*args)
        return call_result
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


async def calculator(expression: str) -> dict[str, Any]:
    """Safely evaluate an arithmetic expression.

    Args:
        expression: Math expression string, e.g. "sqrt(16) + 2 * pi".

    Returns:
        ``{"ok": True, "result": float, "expression": str}`` or error dict.
    """
    expr = expression.strip()
    if not expr:
        return err("Expression is empty", code="EMPTY_EXPRESSION")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        return err(f"Syntax error: {exc}", code="SYNTAX_ERROR")
    try:
        value = _eval_node(tree.body)
        return {"ok": True, "result": float(value), "expression": expression}
    except ZeroDivisionError:
        return err("Division by zero", code="DIVISION_BY_ZERO")
    except (ValueError, TypeError) as exc:
        return err(str(exc), code="EVAL_ERROR")
