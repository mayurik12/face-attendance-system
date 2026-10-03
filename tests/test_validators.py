import pytest

from utils.errors import ValidationError
from utils.validators import validate_department, validate_email, validate_name, validate_user_code


@pytest.mark.parametrize("bad", ["", "   ", None, "123", "Bob<script>", "x" * 200])
def test_invalid_names(bad):
    with pytest.raises(ValidationError):
        validate_name(bad)


def test_name_is_normalised():
    assert validate_name("  Asha   K. Patil ") == "Asha K. Patil"
    assert validate_name("O'Brien-Smith") == "O'Brien-Smith"


@pytest.mark.parametrize("bad", ["", " ", None, "a", "has space", "id/with/slash", "x" * 40, "-lead"])
def test_invalid_ids(bad):
    with pytest.raises(ValidationError):
        validate_user_code(bad)


def test_valid_id():
    assert validate_user_code(" MCA2024_01 ") == "MCA2024_01"


def test_email_optional_and_validated():
    assert validate_email("") is None
    assert validate_email("a@b.co") == "a@b.co"
    with pytest.raises(ValidationError):
        validate_email("not-an-email")


def test_department():
    assert validate_department("  ") is None
    assert validate_department("Computer  Science") == "Computer Science"
    with pytest.raises(ValidationError):
        validate_department("d" * 101)
