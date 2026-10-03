from datetime import date, datetime

import numpy as np
import pytest

from database.db import Database
from tests.conftest import make_embedding
from utils.errors import DatabaseError, DuplicateUserError, ValidationError


def add(db, code="S001", name="Asha Rao", seeds=(1, 2)):
    return db.add_user_with_embeddings(code, name, None, "CS", [make_embedding(s) for s in seeds])


def test_creates_tables(db):
    import sqlite3
    tables = {r[0] for r in sqlite3.connect(db.path).execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"users", "face_embeddings", "attendance"} <= tables


def test_insert_and_fetch_user(db):
    user = add(db)
    assert user.id > 0 and user.name == "Asha Rao"
    assert db.get_user_by_code("S001").id == user.id
    assert db.count_embeddings(user.id) == 2
    assert len(db.list_users()) == 1


def test_duplicate_id_rejected_and_nothing_partial_saved(db):
    add(db)
    with pytest.raises(DuplicateUserError):
        add(db, name="Someone Else")
    assert len(db.list_users()) == 1
    assert db.load_all_embeddings()[1].shape[0] == 2


def test_embeddings_round_trip(db):
    original = make_embedding(5)
    user = db.add_user_with_embeddings("S9", "Ravi", None, None, [original])
    ids, matrix = db.load_all_embeddings()
    assert ids.tolist() == [user.id]
    assert np.allclose(matrix[0], original)


def test_invalid_embedding_rejected(db):
    with pytest.raises(ValidationError):
        db.add_user_with_embeddings("S1", "Bad", None, None, [np.zeros(10)])
    with pytest.raises(ValidationError):
        db.add_user_with_embeddings("S1", "Bad", None, None, [])
    assert db.list_users() == []


def test_delete_user_cascades(db):
    user = add(db)
    db.add_attendance(user.id, datetime(2025, 1, 1, 9, 0), "PRESENT", 0.3)
    assert db.delete_user(user.id) is True
    assert db.delete_user(user.id) is False
    assert db.load_all_embeddings()[1].shape[0] == 0
    assert db.query_attendance() == []


def test_delete_biometrics_keeps_user_and_history(db):
    user = add(db)
    db.add_attendance(user.id, datetime(2025, 1, 1, 9, 0))
    assert db.delete_biometrics(user.id) == 2
    assert db.count_embeddings(user.id) == 0
    assert len(db.list_users()) == 1 and len(db.query_attendance()) == 1


def test_attendance_insert_and_filters(db):
    a, b = add(db, "S001", "Asha Rao"), add(db, "S002", "Ravi Kumar", seeds=(3,))
    db.add_attendance(a.id, datetime(2025, 3, 1, 9, 0, 5), "PRESENT", 0.31)
    db.add_attendance(a.id, datetime(2025, 3, 5, 9, 1, 0), "PRESENT", 0.29)
    db.add_attendance(b.id, datetime(2025, 3, 5, 10, 0, 0), "PRESENT", 0.40)
    assert len(db.query_attendance()) == 3
    assert [r["name"] for r in db.query_attendance(name="ravi")] == ["Ravi Kumar"]
    assert len(db.query_attendance(employee_code="S001")) == 2
    assert len(db.query_attendance(on_date=date(2025, 3, 5))) == 2
    assert len(db.query_attendance(date_from=date(2025, 3, 2), date_to=date(2025, 3, 31))) == 2
    assert len(db.query_attendance(date_to=date(2025, 3, 1))) == 1
    first = db.query_attendance()[0]
    assert first["time"] == "10:00:00" and first["date"] == "2025-03-05"
    assert db.last_attendance_time(a.id) == datetime(2025, 3, 5, 9, 1, 0)


def test_like_wildcards_are_literal(db):
    add(db)
    db.add_attendance(1, datetime(2025, 1, 1, 9, 0))
    assert db.query_attendance(name="%") == []


def test_attendance_requires_existing_user(db):
    with pytest.raises(DatabaseError):
        db.add_attendance(999, datetime.now())


def test_database_failure_is_wrapped(tmp_path):
    bad = Database(tmp_path)  # a directory cannot be opened as a database
    with pytest.raises(DatabaseError):
        bad.initialize()


def test_missing_schema(tmp_path):
    with pytest.raises(DatabaseError):
        Database(tmp_path / "x.db", schema_path=tmp_path / "nope.sql").initialize()


def test_corrupt_database_file(tmp_path):
    path = tmp_path / "corrupt.db"
    path.write_bytes(b"this is not a sqlite database" * 50)
    with pytest.raises(DatabaseError):
        Database(path).initialize()
