import doctest
import unittest
import utils


def run_doctests() -> None:
    res = doctest.testmod(utils)
    if res.failed:
        raise SystemExit(f"Doctest failures: {res.failed}")


def run_unittests() -> None:
    loader = unittest.TestLoader()
    suite = loader.discover("tests")
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    if not res.wasSuccessful():
        raise SystemExit("Unit tests failed")


if __name__ == "__main__":
    run_doctests()
    run_unittests()
