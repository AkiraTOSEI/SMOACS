import argparse

import pytest

from src.utils.config_loader import (
    add_force_pressure_arguments,
    build_force_pressure_config,
)


def parse(argv=()):
    parser = argparse.ArgumentParser()
    add_force_pressure_arguments(parser)
    return build_force_pressure_config(parser.parse_args(argv))


def test_defaults_and_independent_switches():
    config = parse()
    assert config == dict(
        use_force_loss=False,
        use_pressure_loss=False,
        force_loss_mode="log",
        pressure_loss_mode="log",
        F_ref=1.0,
        P_ref=1.0,
        force_loss_clip=1e-7,
        pressure_loss_clip=1e-7,
    )
    assert parse(["--use_force_loss"])["use_force_loss"]
    assert not parse(["--use_force_loss"])["use_pressure_loss"]


@pytest.mark.parametrize("force", ["log", "raw"])
@pytest.mark.parametrize("pressure", ["log", "raw"])
def test_modes_and_argument_forms(force, pressure):
    config = parse(
        [
            f"--force_loss_mode={force}",
            "--pressure_loss_mode",
            pressure,
            "--F_ref=2",
            "--P_ref",
            "3",
            "--force_loss_clip=0.01",
        ]
    )
    assert config["force_loss_mode"] == force
    assert config["pressure_loss_mode"] == pressure
    assert config["F_ref"] == 2 and config["P_ref"] == 3
    assert config["force_loss_clip"] == 0.01
    assert config["pressure_loss_clip"] == 1e-7


@pytest.mark.parametrize(
    "key", ["F_ref", "P_ref", "force_loss_clip", "pressure_loss_clip"]
)
@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "-1"])
def test_invalid_numbers(key, value):
    with pytest.raises(ValueError):
        parse([f"--{key}={value}"])


@pytest.mark.parametrize("key", ["force_loss_clip", "pressure_loss_clip"])
def test_zero_clip_rejected_in_raw_mode(key):
    with pytest.raises(ValueError):
        parse([f"--{key}=0", "--force_loss_mode=raw", "--pressure_loss_mode=raw"])


def test_unknown_mode_and_zero_weights():
    with pytest.raises(SystemExit):
        parse(["--force_loss_mode=unknown"])
    assert parse(["--F_ref=0"])["F_ref"] == 0


@pytest.mark.parametrize(
    "arguments",
    [["--mattersim_model", "explicit.pth"], ["--mattersim_model=explicit.pth"], []],
)
def test_actual_main_model_precedence_without_launching_search(arguments):
    import ast
    from pathlib import Path
    from types import SimpleNamespace

    # Execute only the existing energy-config resolution block from the CLI entrypoint.
    tree = ast.parse((Path(__file__).parents[2] / "main.py").read_text())
    block = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "config and 'energy' in config"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--mattersim_model")
    args = parser.parse_args(arguments)
    namespace = dict(
        config={"energy": {"model": "configured.pth"}},
        args=args,
        mattersim_model=args.mattersim_model,
        sys=SimpleNamespace(argv=arguments),
    )
    exec(
        compile(ast.Module(body=[block], type_ignores=[]), "main.py", "exec"), namespace
    )
    assert namespace["mattersim_model"] == (
        "explicit.pth" if arguments else "configured.pth"
    )
