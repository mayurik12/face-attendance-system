from datetime import datetime, timedelta

from core.attendance import AttendanceManager
from tests.conftest import make_embedding


class FakeClock:
    def __init__(self):
        self.now = datetime(2025, 5, 1, 9, 0, 0)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


def setup(db, cooldown=300):
    user = db.add_user_with_embeddings("S1", "Asha", None, None, [make_embedding(1)])
    clock = FakeClock()
    return user, clock, AttendanceManager(db, cooldown, clock)


def test_first_recognition_marks_attendance(db):
    user, _, mgr = setup(db)
    assert mgr.mark(user.id, 0.3) is not None
    rows = db.query_attendance()
    assert len(rows) == 1 and rows[0]["status"] == "PRESENT" and rows[0]["recognition_distance"] == 0.3


def test_repeat_within_cooldown_is_ignored(db):
    user, clock, mgr = setup(db)
    mgr.mark(user.id)
    for _ in range(50):  # simulates 50 consecutive frames
        clock.advance(1)
        assert mgr.mark(user.id) is None
    assert len(db.query_attendance()) == 1


def test_after_cooldown_marks_again(db):
    user, clock, mgr = setup(db, cooldown=300)
    mgr.mark(user.id)
    clock.advance(300)
    assert mgr.mark(user.id) is not None
    assert len(db.query_attendance()) == 2


def test_cooldown_survives_restart(db):
    user, clock, mgr = setup(db)
    mgr.mark(user.id)
    restarted = AttendanceManager(db, 300, clock)  # empty in-memory cache
    clock.advance(10)
    assert restarted.mark(user.id) is None


def test_cooldown_is_per_user(db):
    user, clock, mgr = setup(db)
    other = db.add_user_with_embeddings("S2", "Ravi", None, None, [make_embedding(2)])
    mgr.mark(user.id)
    assert mgr.mark(other.id) is not None
