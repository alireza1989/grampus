"""Tests for grampus.tools.library.calculator."""

from __future__ import annotations

from grampus.tools.library.calculator import calculator


class TestBasicArithmetic:
    async def test_addition(self) -> None:
        result = await calculator(expression="2 + 3")
        assert result["ok"] is True
        assert result["result"] == 5.0

    async def test_subtraction(self) -> None:
        result = await calculator(expression="10 - 4")
        assert result["ok"] is True
        assert result["result"] == 6.0

    async def test_multiplication(self) -> None:
        result = await calculator(expression="3 * 7")
        assert result["ok"] is True
        assert result["result"] == 21.0

    async def test_division(self) -> None:
        result = await calculator(expression="15 / 4")
        assert result["ok"] is True
        assert result["result"] == 3.75

    async def test_integer_division(self) -> None:
        result = await calculator(expression="15 // 4")
        assert result["ok"] is True
        assert result["result"] == 3.0

    async def test_modulo(self) -> None:
        result = await calculator(expression="17 % 5")
        assert result["ok"] is True
        assert result["result"] == 2.0

    async def test_power(self) -> None:
        result = await calculator(expression="2 ** 8")
        assert result["ok"] is True
        assert result["result"] == 256.0


class TestMathFunctions:
    async def test_sqrt(self) -> None:
        result = await calculator(expression="sqrt(16)")
        assert result["ok"] is True
        assert result["result"] == 4.0

    async def test_abs_negative(self) -> None:
        result = await calculator(expression="abs(-7)")
        assert result["ok"] is True
        assert result["result"] == 7.0

    async def test_round(self) -> None:
        result = await calculator(expression="round(3.14159, 2)")
        assert result["ok"] is True
        assert result["result"] == 3.14

    async def test_pi_constant(self) -> None:
        result = await calculator(expression="pi")
        assert result["ok"] is True
        assert abs(result["result"] - 3.14159) < 0.001

    async def test_e_constant(self) -> None:
        result = await calculator(expression="e")
        assert result["ok"] is True
        assert abs(result["result"] - 2.71828) < 0.001

    async def test_floor(self) -> None:
        result = await calculator(expression="floor(3.9)")
        assert result["ok"] is True
        assert result["result"] == 3.0

    async def test_ceil(self) -> None:
        result = await calculator(expression="ceil(3.1)")
        assert result["ok"] is True
        assert result["result"] == 4.0

    async def test_log(self) -> None:
        result = await calculator(expression="log(1)")
        assert result["ok"] is True
        assert abs(result["result"]) < 1e-10

    async def test_sin(self) -> None:
        result = await calculator(expression="sin(0)")
        assert result["ok"] is True
        assert abs(result["result"]) < 1e-10

    async def test_cos(self) -> None:
        result = await calculator(expression="cos(0)")
        assert result["ok"] is True
        assert result["result"] == 1.0

    async def test_tan(self) -> None:
        result = await calculator(expression="tan(0)")
        assert result["ok"] is True
        assert abs(result["result"]) < 1e-10


class TestErrors:
    async def test_division_by_zero_returns_err(self) -> None:
        result = await calculator(expression="1 / 0")
        assert result["ok"] is False
        assert "error" in result

    async def test_unknown_function_returns_err(self) -> None:
        result = await calculator(expression="evil(42)")
        assert result["ok"] is False
        assert "error" in result

    async def test_empty_expression_returns_err(self) -> None:
        result = await calculator(expression="")
        assert result["ok"] is False
        assert "error" in result

    async def test_does_not_raise(self) -> None:
        # Tools must never raise — errors come back as dicts
        result = await calculator(expression="__import__('os')")
        assert result["ok"] is False

    async def test_result_is_float(self) -> None:
        result = await calculator(expression="7 + 3")
        assert result["ok"] is True
        assert isinstance(result["result"], float)

    async def test_expression_included_in_result(self) -> None:
        result = await calculator(expression="2 + 2")
        assert result["ok"] is True
        assert result["expression"] == "2 + 2"
