import pytest

from sqlquality.dialects import validate_dialect


def test_known_dialect_normalized():
    assert validate_dialect("postgres") == "postgres"
    assert validate_dialect(" Redshift ") == "redshift"
    assert validate_dialect("SNOWFLAKE") == "snowflake"


def test_unknown_dialect_raises_with_suggestions():
    with pytest.raises(ValueError) as excinfo:
        validate_dialect("oracle9000")
    message = str(excinfo.value)
    assert "oracle9000" in message
    assert "postgres" in message


def test_blank_dialect_raises():
    with pytest.raises(ValueError):
        validate_dialect("   ")
