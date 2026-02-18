"""
Tests for LLM code validation utilities.

These tests verify the code validation and fixing functions that
sanitize LLM-generated code before execution.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Fund_Manager'))


class TestValidateSyntax:
    """Tests for validate_syntax function."""

    def test_valid_code(self):
        """Should return True for valid Python code."""
        from code_validator import validate_syntax

        code = """
import pandas as pd

def test():
    return 42
"""
        is_valid, error = validate_syntax(code)

        assert is_valid is True
        assert error is None

    def test_invalid_syntax(self):
        """Should return False with error for invalid syntax."""
        from code_validator import validate_syntax

        code = """
def test()
    return 42
"""
        is_valid, error = validate_syntax(code)

        assert is_valid is False
        assert "Syntax error" in error

    def test_empty_code(self):
        """Should handle empty code."""
        from code_validator import validate_syntax

        is_valid, error = validate_syntax("")

        assert is_valid is True


class TestExtractImports:
    """Tests for extract_imports function."""

    def test_import_statement(self):
        """Should extract from import statements."""
        from code_validator import extract_imports

        code = "import pandas"
        imports = extract_imports(code)

        assert "pandas" in imports

    def test_import_as(self):
        """Should extract base module from 'as' imports."""
        from code_validator import extract_imports

        code = "import pandas as pd"
        imports = extract_imports(code)

        assert "pandas" in imports

    def test_from_import(self):
        """Should extract from 'from X import Y' statements."""
        from code_validator import extract_imports

        code = "from datetime import time"
        imports = extract_imports(code)

        assert "datetime" in imports

    def test_multiple_imports(self):
        """Should extract all imports."""
        from code_validator import extract_imports

        code = """
import pandas as pd
import numpy as np
from datetime import datetime, time
from queue import Queue
"""
        imports = extract_imports(code)

        assert "pandas" in imports
        assert "numpy" in imports
        assert "datetime" in imports
        assert "queue" in imports


class TestCheckDatetimeImport:
    """Tests for check_datetime_import function."""

    def test_correct_import(self):
        """Should pass when datetime.time properly imported."""
        from code_validator import check_datetime_import

        code = """
from datetime import time

x = time(9, 30)
"""
        is_valid, fixed = check_datetime_import(code)

        assert is_valid is True

    def test_wrong_import_time(self):
        """Should fix when 'import time' used with time(h, m)."""
        from code_validator import check_datetime_import

        code = """
import time

x = time(9, 30)
"""
        is_valid, fixed = check_datetime_import(code)

        assert is_valid is False
        assert "from datetime import time" in fixed

    def test_no_time_constructor(self):
        """Should pass when no time constructor used."""
        from code_validator import check_datetime_import

        code = """
import time

time.sleep(1)
"""
        is_valid, _ = check_datetime_import(code)

        assert is_valid is True


class TestValidateRequiredImports:
    """Tests for validate_required_imports function."""

    def test_missing_pandas(self, missing_imports_code):
        """Should detect missing pandas import."""
        from code_validator import validate_required_imports

        is_valid, missing = validate_required_imports(missing_imports_code)

        assert is_valid is False
        assert "pandas" in missing

    def test_all_present(self, valid_strategy_code):
        """Should pass when all required imports present."""
        from code_validator import validate_required_imports

        is_valid, missing = validate_required_imports(valid_strategy_code)

        assert is_valid is True
        assert len(missing) == 0


class TestInjectMissingImports:
    """Tests for inject_missing_imports function."""

    def test_add_pandas(self):
        """Should add pandas import."""
        from code_validator import inject_missing_imports

        code = "x = pd.DataFrame()"
        fixed = inject_missing_imports(code, ["pandas"])

        assert "import pandas as pd" in fixed
        assert fixed.startswith("import pandas")

    def test_add_numpy(self):
        """Should add numpy import."""
        from code_validator import inject_missing_imports

        code = "x = np.array([1, 2])"
        fixed = inject_missing_imports(code, ["numpy"])

        assert "import numpy as np" in fixed

    def test_add_datetime(self):
        """Should add datetime import."""
        from code_validator import inject_missing_imports

        code = "x = time(9, 30)"
        fixed = inject_missing_imports(code, ["datetime"])

        assert "from datetime import datetime, time" in fixed

    def test_add_multiple(self):
        """Should add multiple imports."""
        from code_validator import inject_missing_imports

        code = "pass"
        fixed = inject_missing_imports(code, ["pandas", "numpy", "queue"])

        assert "import pandas" in fixed
        assert "import numpy" in fixed
        assert "from queue import Queue" in fixed


class TestRemoveMarkdownArtifacts:
    """Tests for remove_markdown_artifacts function."""

    def test_removes_code_blocks(self):
        """Should remove markdown code block markers."""
        from code_validator import remove_markdown_artifacts

        code = """```python
import pandas as pd
```"""
        cleaned = remove_markdown_artifacts(code)

        assert "```" not in cleaned
        assert "import pandas" in cleaned

    def test_removes_preamble(self):
        """Should remove LLM preamble text."""
        from code_validator import remove_markdown_artifacts

        code = """Here's the modified code:

import pandas as pd
"""
        cleaned = remove_markdown_artifacts(code)

        assert "Here's" not in cleaned
        assert "import pandas" in cleaned

    def test_removes_trailing_explanation(self):
        """Should remove trailing explanations."""
        from code_validator import remove_markdown_artifacts

        code = """import pandas as pd

This code does X, Y, Z..."""
        cleaned = remove_markdown_artifacts(code)

        assert "This code does" not in cleaned


class TestValidateAndFixCode:
    """Tests for the full validation pipeline."""

    def test_valid_code_passes(self, valid_strategy_code):
        """Should pass valid code through unchanged (except whitespace)."""
        from code_validator import validate_and_fix_code

        is_valid, fixed, issues = validate_and_fix_code(valid_strategy_code)

        assert is_valid is True

    def test_fixes_syntax_error_returns_false(self, invalid_syntax_code):
        """Should return False for unfixable syntax errors."""
        from code_validator import validate_and_fix_code

        is_valid, fixed, issues = validate_and_fix_code(invalid_syntax_code)

        assert is_valid is False
        assert any("Syntax error" in issue for issue in issues)

    def test_fixes_missing_imports(self, missing_imports_code):
        """Should inject missing imports."""
        from code_validator import validate_and_fix_code

        is_valid, fixed, issues = validate_and_fix_code(missing_imports_code)

        # Should add missing pandas import
        assert "import pandas" in fixed
        assert any("missing imports" in issue for issue in issues)

    def test_cleans_markdown(self):
        """Should remove markdown artifacts."""
        from code_validator import validate_and_fix_code

        code = """```python
import pandas as pd
x = 1
```"""
        is_valid, fixed, issues = validate_and_fix_code(code)

        assert is_valid is True
        assert "```" not in fixed
        assert any("markdown" in issue.lower() for issue in issues)


class TestCreateStrategyTemplate:
    """Tests for create_strategy_template function."""

    def test_includes_required_imports(self):
        """Should include all required imports."""
        from code_validator import create_strategy_template

        template = create_strategy_template()

        assert "import pandas as pd" in template
        assert "import numpy as np" in template
        assert "from datetime import datetime, time" in template
        assert "from queue import Queue" in template

    def test_custom_class_name(self):
        """Should use custom class name."""
        from code_validator import create_strategy_template

        template = create_strategy_template(class_name="MyCustomStrategy")

        assert "class MyCustomStrategy" in template

    def test_has_calculate_signals(self):
        """Should have calculate_signals method stub."""
        from code_validator import create_strategy_template

        template = create_strategy_template()

        assert "def calculate_signals" in template


class TestGetValidationPromptSuffix:
    """Tests for get_validation_prompt_suffix function."""

    def test_returns_instructions(self):
        """Should return code requirements."""
        from code_validator import get_validation_prompt_suffix

        suffix = get_validation_prompt_suffix()

        assert "import pandas" in suffix
        assert "from datetime import" in suffix
        assert "time(9, 30)" in suffix
