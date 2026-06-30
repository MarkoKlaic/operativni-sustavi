import ast
import os
from typing import List, Tuple

ROOT = os.path.dirname(os.path.dirname(__file__))


def files_py(root: str) -> List[str]:
    out = []
    for dirpath, dirs, files in os.walk(root):
        if any(
            part.startswith(".") or part == "__pycache__"
            for part in dirpath.split(os.sep)
        ):
            continue
        for f in files:
            if f.endswith(".py"):
                out.append(os.path.join(dirpath, f))
    return out


def check_file(path: str) -> Tuple[List[str], List[str]]:
    long_lines: List[str] = []
    missing_hints: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()
    for i, line in enumerate(src.splitlines(), start=1):
        if len(line) > 88:
            msg = f"{path}:{i}: line too long ({len(line)} > 88)"
            long_lines.append(msg)
        if line.rstrip() != line:
            msg = f"{path}:{i}: trailing whitespace"
            long_lines.append(msg)
        if "\t" in line:
            msg = f"{path}:{i}: tab character used"
            long_lines.append(msg)

    try:
        tree = ast.parse(src, filename=path)
    except SyntaxError as e:
        long_lines.append(f"{path}: syntax error: {e}")
        return long_lines, missing_hints

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            if name.startswith("_"):
                continue
            has_missing = False
            for arg in node.args.args:
                if arg.annotation is None and arg.arg != "self":
                    has_missing = True
            if node.returns is None:
                has_missing = True
            if has_missing:
                msg = f"{path}:{node.lineno}: function '{name}'" " missing type hints"
                missing_hints.append(msg)

    return long_lines, missing_hints


def main() -> None:
    pyfiles = files_py(ROOT)
    all_long = []
    all_missing = []
    for p in pyfiles:
        l, m = check_file(p)
        all_long.extend(l)
        all_missing.extend(m)

    if not all_long and not all_missing:
        msg = (
            "No basic style/type issues found "
            "(line-length/trailing/tab or missing hints)."
        )
        print(msg)
        return
    if all_long:
        print("Line/format issues:")
        for s in all_long:
            print("  ", s)
    if all_missing:
        print("\nMissing type hints:")
        for s in all_missing:
            print("  ", s)


if __name__ == "__main__":
    main()
