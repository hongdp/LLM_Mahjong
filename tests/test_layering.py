"""Dependency direction (CLAUDE.md project map): library code never reaches up.

    src      must not import  scripts / tests
    scripts  must not import  tests

Checked statically (AST, including imports nested inside functions), so a lazy
`from scripts.x import y` inside a method is caught as well.

One import is grandfathered: `src/tasks/mahjong/hanchan.py` is content-
fingerprinted (run_elo_league.HANCHAN_FILES, src/eval/fingerprints.SURFACES);
editing a byte of it voids the hanchan rating pool, so its lazy import of
`scripts.train_placement_value.PlacementValue` stays until the next epoch
recalibration. That script re-exports the class from
`src/agents/dnn/placement_value.py`, which this test also pins.
"""
import ast
import glob
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GRANDFATHERED = {
    ("src/tasks/mahjong/hanchan.py", "scripts.train_placement_value"),
}


def _imports(path):
    """Yield every absolute module name imported anywhere in the file."""
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module


def _violations(pattern, forbidden_roots):
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, pattern), recursive=True)):
        rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
        for mod in _imports(path):
            if mod.split(".")[0] in forbidden_roots and (rel, mod) not in GRANDFATHERED:
                out.append(f"{rel} imports {mod}")
    return out


class LayeringTest(unittest.TestCase):
    def test_src_does_not_import_scripts_or_tests(self):
        self.assertEqual(_violations("src/**/*.py", {"scripts", "tests"}), [])

    def test_scripts_do_not_import_tests(self):
        self.assertEqual(_violations("scripts/**/*.py", {"tests"}), [])

    def test_grandfathered_imports_still_exist(self):
        # a stale allowlist entry hides nothing but should be removed when the
        # frozen file is finally edited
        for rel, mod in GRANDFATHERED:
            self.assertIn(mod, set(_imports(os.path.join(ROOT, rel))), (rel, mod))

    def test_frozen_import_resolves_to_the_src_class(self):
        from scripts.train_placement_value import PlacementValue as via_script
        from src.agents.dnn.placement_value import PlacementValue
        self.assertIs(via_script, PlacementValue)


if __name__ == "__main__":
    unittest.main()
