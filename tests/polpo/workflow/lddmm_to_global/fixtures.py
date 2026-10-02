from pathlib import Path

import pytest

from polpo.workflow.lddmm_to_global import LddmmToGlobalOutput


@pytest.fixture(scope="session")
def lddmm_output():
    lddmm_output_dir = (Path("tests") / ".test_data" / "lddmm").resolve()

    if not lddmm_output_dir.exists():
        pytest.fail(
            "LDDMM test output is missing. "
            "Generate it with `python -m tests.generate_lddmm_output`."
        )

    return LddmmToGlobalOutput(lddmm_output_dir)
