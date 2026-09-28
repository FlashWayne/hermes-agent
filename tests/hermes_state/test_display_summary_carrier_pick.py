"""The display projection must show the plain user prompt, not its summary carrier (#126102).

A force-user-leading compaction commits the summary as an ``active = 1`` user row whose
content is ``summary + END MARKER + the user's ask`` (the carrier the model needs). The
carrier's display identity normalizes back to the live ask, so it lands in the SAME
``display_order`` group as the durable plain prompt, and the group pick
(``ORDER BY candidate.active DESC, candidate.id DESC``) chose the carrier — the desktop
hydrated it as a normal bubble and showed ``[CONTEXT COMPACTION SUMMARY]`` instead of the
message the user just sent.

Invariant: within one display slot, a plain prompt outranks a summary carrier; and a
compaction commit must not rewind-archive the plain original out of the display when the
surviving copy IS its summary carrier (the plain row is the only clean render of that turn).
"""

from agent.context_compressor import SUMMARY_PREFIX, _SUMMARY_END_MARKER
from hermes_state import SessionDB

CARRIER_CONTENT = (
    SUMMARY_PREFIX + "\nEarlier turns were compacted.\n\n" + _SUMMARY_END_MARKER + "\n\nfix the freeze please"
)


def _user_texts(db, sid):
    return [m["content"] for m in db.get_messages(sid, include_compacted=True) if m["role"] == "user"]


class TestDisplayPrefersPlainPromptOverSummaryCarrier:
    def test_pick_prefers_plain_prompt_in_a_carrier_group(self, tmp_path):
        db = SessionDB(tmp_path / "state.db")
        sid = "20260927_230710_carrierpick"
        db.create_session(sid, "cli", model="test/model")
        ts = 1727000000.0
        # The plain prompt and its compaction carrier share one display identity (the
        # carrier's content normalizes back to the live ask), so the insert trigger
        # folds them into one display_order group.
        db.append_message(sid, "user", "fix the freeze please", timestamp=ts)
        db.append_message(sid, "user", CARRIER_CONTENT, timestamp=ts)
        db.append_message(sid, "assistant", "on it", timestamp=ts + 1)
        # Durable state the compaction pipeline leaves behind: the carrier is the
        # active row; the plain original is archived display-visible.
        db._execute_write(
            lambda conn: conn.execute("UPDATE messages SET active = 0, compacted = 1 WHERE id = 1"))

        texts = _user_texts(db, sid)

        assert "fix the freeze please" in texts
        assert not any(_SUMMARY_END_MARKER in (t or "") for t in texts)

    def test_compaction_commit_keeps_the_plain_prompt_displayable(self, tmp_path):
        db = SessionDB(tmp_path / "state.db")
        sid = "20260927_230710_carriercommit"
        db.create_session(sid, "cli", model="test/model")
        db.append_message(sid, "user", "fix the freeze please", timestamp=1727000000.0)
        db.append_message(sid, "assistant", "on it", timestamp=1727000001.0)
        held = db.get_messages_as_conversation(sid, include_row_ids=True)
        # The carrier's summary-prefixed content matches no durable row, so the carried
        # original stays unresolved — the archive path that produced the issue's rows.
        carrier = {"role": "user", "content": CARRIER_CONTENT, "timestamp": 1727000000.0}
        assistant_copy = {"role": "assistant", "content": "on it", "timestamp": 1727000001.0}

        db.archive_and_compact(
            sid, [carrier, assistant_copy], carried_messages=[carrier, assistant_copy],
            watermark=db.get_active_message_watermark(sid), covered_ids=[1, 2], unresolved_held=[])

        # The plain prompt's durable row is still display-visible (not rewind-archived
        # out of the display by its own summary carrier winning the identity group).
        rows = db._read_all(
            "SELECT role, content, active, compacted FROM messages WHERE session_id = ? AND id = 1", (sid,))
        assert rows and rows[0]["active"] == 0 and rows[0]["compacted"] == 1

        texts = _user_texts(db, sid)
        # One display slot: the plain ask, exactly once, never the summary scaffolding.
        assert texts == ["fix the freeze please"]
