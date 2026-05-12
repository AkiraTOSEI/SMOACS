import ast
from typing import Any, Callable, Dict

import torch


class SafeExpressionError(ValueError):
    pass


def safe_div(a, b, eps: float = 1e-8):
    return a / (b + eps)


def clamp(x, min_val: float = None, max_val: float = None):
    return torch.clamp(x, min=min_val, max=max_val)


def mse(a, b):
    return (a - b) ** 2


def mae(a, b):
    return torch.abs(a - b)

def relu(x):
    return torch.relu(x)


def _promote_tensors(a, b):
    if torch.is_tensor(a) and not torch.is_tensor(b):
        b = torch.as_tensor(b, device=a.device, dtype=a.dtype)
    elif torch.is_tensor(b) and not torch.is_tensor(a):
        a = torch.as_tensor(a, device=b.device, dtype=b.dtype)
    return a, b


def tensor_max(a, b):
    a, b = _promote_tensors(a, b)
    return torch.maximum(a, b)


def tensor_min(a, b):
    a, b = _promote_tensors(a, b)
    return torch.minimum(a, b)


class SafeExpressionEvaluator:
    def __init__(self, allowed_functions: Dict[str, Callable[..., Any]] = None) -> None:
        default_funcs = {
            "safe_div": safe_div,
            "abs": torch.abs,
            "min": tensor_min,
            "max": tensor_max,
            "log": torch.log,
            "exp": torch.exp,
            "clamp": clamp,
            "mse": mse,
            "mae": mae,
            "relu": relu,
            "round": torch.round,
        }
        self.allowed_functions = default_funcs
        if allowed_functions:
            self.allowed_functions.update(allowed_functions)

    def eval(self, expr: str, variables: Dict[str, Any]) -> Any:
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            raise SafeExpressionError(f"Invalid expression: {expr}") from exc
        return self._eval_node(tree.body, variables)

    def _eval_node(self, node: ast.AST, variables: Dict[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in variables:
                return variables[node.id]
            raise SafeExpressionError(f"Unknown variable: {node.id}")
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "pred":
                pred_dict = variables.get("pred")
                if pred_dict is None:
                    raise SafeExpressionError("pred is not provided")
                if node.attr not in pred_dict:
                    raise SafeExpressionError(f"Unknown pred key: {node.attr}")
                return pred_dict[node.attr]
            raise SafeExpressionError("Only pred.<name> attribute access is allowed")
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, variables)
            right = self._eval_node(node.right, variables)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            if isinstance(node.op, ast.Pow):
                return left ** right
            raise SafeExpressionError("Unsupported binary operator")
        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand, variables)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise SafeExpressionError("Unsupported unary operator")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            else:
                raise SafeExpressionError("Only direct function calls are allowed")
            if func_name in self.allowed_functions:
                func = self.allowed_functions[func_name]
            else:
                func = variables.get(func_name)
                if func is None or not callable(func):
                    raise SafeExpressionError(f"Function not allowed: {func_name}")
            args = [self._eval_node(arg, variables) for arg in node.args]
            return func(*args)
        raise SafeExpressionError(f"Unsupported expression node: {type(node).__name__}")
