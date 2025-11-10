"""A very small testing framework compatible with a subset of pytest's API.

This module only implements the pieces that our test-suite relies on:

* ``pytest.approx`` for floating point comparisons.
* ``pytest.raises`` as a context manager.
* ``pytest.fixture`` decorator with basic dependency injection and
  ``autouse`` support.
* A ``monkeypatch`` fixture that can patch attributes during a test and
  automatically restore them afterwards.
* A minimal test runner exposed through ``pytest.main`` which can execute
  test modules discovered under the provided paths.

It is intentionally tiny but keeps the familiar ``pytest`` API surface so
we can depend on ``poetry run pytest`` without pulling the real pytest
package into our isolated execution environment.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import os
import re
import sys
import traceback
from contextlib import ContextDecorator
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Sequence

__all__ = [
    "approx",
    "fixture",
    "raises",
    "main",
    "mark",
]


class Approx:
    """Simplified numeric approximation helper used in assertions."""

    def __init__(self, expected: float, *, rel: float = 1e-12, abs: float = 0.0) -> None:
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def _is_close(self, actual: float) -> bool:
        tolerance = max(self.abs, abs(self.expected) * self.rel)
        return abs(actual - self.expected) <= tolerance

    def __eq__(self, other: object) -> bool:  # pragma: no cover - trivial
        if isinstance(other, (int, float)):
            return self._is_close(float(other))
        return NotImplemented

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Approx(expected={self.expected!r}, rel={self.rel}, abs={self.abs})"


def approx(expected: float, *, rel: float = 1e-12, abs: float = 0.0) -> Approx:
    return Approx(expected, rel=rel, abs=abs)


class RaisesContext(ContextDecorator):
    """Context manager asserting that an exception was raised."""

    def __init__(self, exception_type: type[BaseException], match: Optional[str] = None) -> None:
        self.exception_type = exception_type
        self.match = match
        self.caught: Optional[BaseException] = None

    def __enter__(self) -> "RaisesContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            raise AssertionError(f"Expected {self.exception_type.__name__} to be raised")
        if not issubclass(exc_type, self.exception_type):
            return False
        self.caught = exc
        if self.match and not re.search(self.match, str(exc)):
            raise AssertionError(
                f"Exception message {exc!r} does not match pattern {self.match!r}"
            )
        return True


def raises(exception_type: type[BaseException], *, match: Optional[str] = None) -> RaisesContext:
    return RaisesContext(exception_type, match=match)


@dataclass
class FixtureDefinition:
    func: Callable[..., Any]
    autouse: bool = False
    teardown: Optional[Callable[[Any], None]] = None

    def dependencies(self) -> Sequence[str]:
        return tuple(inspect.signature(self.func).parameters)

    def __call__(self, context: "FixtureContext") -> Any:
        kwargs = {name: context.get_fixture(name) for name in self.dependencies()}
        result = self.func(**kwargs)
        if inspect.isgenerator(result):
            generator = result

            def _finalize(gen: Any = generator) -> None:
                try:
                    next(gen)
                except StopIteration:
                    pass

            try:
                value = next(generator)
            except StopIteration:
                value = None
            context.add_finalizer(_finalize)
            result = value
        if self.teardown is not None:
            context.add_finalizer(lambda: self.teardown(result))
        return result


class MonkeyPatch:
    def __init__(self) -> None:
        self._undo: List[Callable[[], None]] = []

    def setattr(self, target: Any, name: str, value: Any) -> None:
        exists = hasattr(target, name)
        original = getattr(target, name, None)

        def revert() -> None:
            if exists:
                setattr(target, name, original)
            elif hasattr(target, name):
                delattr(target, name)

        setattr(target, name, value)
        self._undo.append(revert)

    def delattr(self, target: Any, name: str) -> None:
        exists = hasattr(target, name)
        original = getattr(target, name, None)

        def revert() -> None:
            if exists:
                setattr(target, name, original)
            elif hasattr(target, name):
                delattr(target, name)

        if exists:
            delattr(target, name)
        self._undo.append(revert)

    def setitem(self, mapping: Any, key: Any, value: Any) -> None:
        exists = key in mapping
        original = mapping[key] if exists else None

        def revert() -> None:
            if exists:
                mapping[key] = original
            elif key in mapping:
                del mapping[key]

        mapping[key] = value
        self._undo.append(revert)

    def delitem(self, mapping: Any, key: Any, raising: bool = True) -> None:
        exists = key in mapping
        if not exists:
            if raising:
                raise KeyError(key)
        original = mapping[key] if exists else None

        def revert() -> None:
            if exists:
                mapping[key] = original
            elif key in mapping:
                del mapping[key]

        if exists:
            del mapping[key]
        self._undo.append(revert)

    def setenv(self, name: str, value: str) -> None:
        exists = name in os.environ
        original = os.environ.get(name)

        def revert() -> None:
            if exists and original is not None:
                os.environ[name] = original
            elif exists and original is None:
                os.environ.pop(name, None)
            else:
                os.environ.pop(name, None)

        os.environ[name] = str(value)
        self._undo.append(revert)

    def delenv(self, name: str, raising: bool = True) -> None:
        exists = name in os.environ
        if not exists and raising:
            raise KeyError(name)
        original = os.environ.get(name)

        def revert() -> None:
            if original is not None:
                os.environ[name] = original
            else:
                os.environ.pop(name, None)

        if exists:
            os.environ.pop(name, None)
        self._undo.append(revert)

    def undo(self) -> None:
        while self._undo:
            undo = self._undo.pop()
            try:
                undo()
            except Exception:
                traceback.print_exc()


class FixtureContext:
    def __init__(self, manager: "FixtureManager") -> None:
        self._manager = manager
        self._cache: Dict[str, Any] = {}
        self._finalizers: List[Callable[[], None]] = []

    def get_fixture(self, name: str) -> Any:
        if name in self._cache:
            return self._cache[name]
        definition = self._manager.resolve(name)
        value = definition(self)
        self._cache[name] = value
        return value

    def add_finalizer(self, callback: Callable[[], None]) -> None:
        self._finalizers.append(callback)

    def finish(self) -> None:
        while self._finalizers:
            callback = self._finalizers.pop()
            try:
                callback()
            except Exception:
                traceback.print_exc()


class FixtureManager:
    def __init__(self, module_name: str) -> None:
        module_fixtures = _fixture_registry.get(module_name, {})
        self._fixtures: Dict[str, FixtureDefinition] = {**_builtin_fixtures, **module_fixtures}
        self._autouse = [name for name, fixture in self._fixtures.items() if fixture.autouse]

    def resolve(self, name: str) -> FixtureDefinition:
        try:
            return self._fixtures[name]
        except KeyError:  # pragma: no cover - defensive
            raise KeyError(f"Unknown fixture '{name}'")

    @property
    def autouse(self) -> Sequence[str]:
        return self._autouse


_fixture_registry: Dict[str, Dict[str, FixtureDefinition]] = {}


def fixture(
    func: Optional[Callable[..., Any]] = None,
    *,
    autouse: bool = False,
    scope: Optional[str] = None,
    name: Optional[str] = None,
    params: Optional[Sequence[Any]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    _ = scope  # scopes other than function are not supported but accepted.
    _ = params  # parameterized fixtures are not supported.

    def decorator(target: Callable[..., Any]) -> Callable[..., Any]:
        module = target.__module__
        registry = _fixture_registry.setdefault(module, {})
        fixture_name = name or target.__name__
        registry[fixture_name] = FixtureDefinition(target, autouse=autouse)
        return target

    if func is not None:
        return decorator(func)
    return decorator


_builtin_fixtures: Dict[str, FixtureDefinition] = {}


def _register_builtin_fixture(name: str, func: Callable[..., Any], *, teardown: Optional[Callable[[Any], None]] = None) -> None:
    _builtin_fixtures[name] = FixtureDefinition(func, autouse=False, teardown=teardown)


def _monkeypatch_fixture() -> MonkeyPatch:
    return MonkeyPatch()


_register_builtin_fixture("monkeypatch", _monkeypatch_fixture, teardown=lambda mp: mp.undo())


def _normalize_argnames(argnames: Any) -> List[str]:
    if isinstance(argnames, str):
        parts = [name.strip() for name in argnames.split(",")]
        return [name for name in parts if name]
    return [str(name) for name in argnames]


def _expand_param_cases(meta: Sequence[tuple[Sequence[str], Sequence[Any], Optional[Sequence[str]]]]) -> List[tuple[Dict[str, Any], List[str]]]:
    cases: List[tuple[Dict[str, Any], List[str]]] = [({}, [])]
    for names, values, ids in meta:
        new_cases: List[tuple[Dict[str, Any], List[str]]] = []
        id_list = list(ids) if ids is not None else None
        for index, value in enumerate(values):
            if isinstance(value, dict):
                assignment = dict(value)
            elif len(names) == 1:
                assignment = {names[0]: value}
            else:
                if not isinstance(value, (list, tuple)):
                    raise TypeError("Parameterized test values must be iterable when multiple arguments are provided")
                if len(value) != len(names):
                    raise ValueError("Parameterized values length does not match argument names")
                assignment = {name: value[idx] for idx, name in enumerate(names)}

            label = id_list[index] if id_list and index < len(id_list) else ", ".join(
                f"{key}={assignment[key]!r}" for key in names if key in assignment
            )

            for base_mapping, base_labels in cases:
                combined = dict(base_mapping)
                combined.update(assignment)
                new_cases.append((combined, base_labels + [label]))
        cases = new_cases
    return cases or [({}, [])]


def _format_case_label(labels: List[str]) -> str:
    cleaned = [label for label in labels if label]
    if not cleaned:
        return ""
    return "[" + " | ".join(cleaned) + "]"


class _Mark:
    def parametrize(
        self,
        argnames: Any,
        argvalues: Sequence[Any],
        ids: Optional[Sequence[str]] = None,
        indirect: bool = False,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        if indirect:
            raise NotImplementedError("indirect parametrization is not supported in this lightweight runner")
        names = _normalize_argnames(argnames)
        values = list(argvalues)

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            existing: List[tuple[Sequence[str], Sequence[Any], Optional[Sequence[str]]]] = getattr(func, "__pytest_params__", [])
            existing.append((names, values, ids))
            setattr(func, "__pytest_params__", existing)
            return func

        return decorator


mark = _Mark()


def _is_test_function(name: str, value: Any) -> bool:
    return name.startswith("test_") and callable(value)


def _import_module_from_path(path: Path) -> ModuleType:
    module_name = "pytest_module_" + "_".join(path.with_suffix("").parts)
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Cannot import test module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _discover_paths(targets: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for target in targets:
        path = Path(target)
        if path.is_dir():
            for root, _, files in os.walk(path):
                for filename in files:
                    if filename.startswith("test_") and filename.endswith(".py"):
                        paths.append(Path(root) / filename)
        elif path.is_file() and path.name.endswith(".py"):
            paths.append(path)
    return paths


def _run_test(
    func: Callable[..., Any],
    manager: FixtureManager,
    param_values: Optional[Dict[str, Any]] = None,
) -> tuple[bool, Optional[str]]:
    context = FixtureContext(manager)
    try:
        for name in manager.autouse:
            context.get_fixture(name)
        params = inspect.signature(func).parameters
        kwargs: Dict[str, Any] = {}
        for name in params:
            if param_values and name in param_values:
                continue
            kwargs[name] = context.get_fixture(name)
        if param_values:
            kwargs.update(param_values)
        func(**kwargs)
        return True, None
    except Exception:
        return False, traceback.format_exc()
    finally:
        context.finish()


def main(args: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="pytest", add_help=False)
    parser.add_argument("paths", nargs="*")
    parser.add_argument("-q", action="store_true", dest="quiet")
    parsed, _ = parser.parse_known_args(args=args)
    targets = parsed.paths or ["tests"]

    test_paths = _discover_paths(targets)
    failures: List[str] = []
    executed = 0

    for path in test_paths:
        module = _import_module_from_path(path)
        manager = FixtureManager(module.__name__)
        for name in dir(module):
            attr = getattr(module, name)
            if not _is_test_function(name, attr):
                continue

            params_meta: List[tuple[Sequence[str], Sequence[Any], Optional[Sequence[str]]]] = getattr(
                attr, "__pytest_params__", []
            )
            case_definitions = _expand_param_cases(params_meta) if params_meta else [({}, [])]

            for param_values, labels in case_definitions:
                executed += 1
                label = _format_case_label(labels)
                success, error = _run_test(attr, manager, param_values or None)
                if not success and error is not None:
                    failures.append(f"{path}:{name}{label}\n{error}")

    if failures:
        if not parsed.quiet:
            for failure in failures:
                print(failure)
        print(f"{len(failures)} failed, {executed - len(failures)} passed")
        return 1

    print(f"{executed} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
