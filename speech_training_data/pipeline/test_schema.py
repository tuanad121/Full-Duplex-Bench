from __future__ import annotations

import unittest

from schema import validate_conversation


def valid_row() -> dict:
    rounds = []
    for index in range(6):
        event = None
        if index == 2:
            event = {"type": "backchannel", "text": "Vâng, mình nghe đây.", "agent_after": "", "overlap_fraction": 0.45}
        if index == 4:
            event = {"type": "real_interruption", "text": "Khoan, đổi sang ngày mai nhé.", "agent_after": "Được rồi, mình sẽ chuyển lịch sang ngày mai cho bạn.", "overlap_fraction": 0.4}
        rounds.append({"user": f"Mình muốn hỏi thông tin chi tiết cho lượt thứ {index + 1}.", "agent": f"Mình sẽ giải thích nội dung lượt thứ {index + 1} thật rõ ràng cho bạn.", "event": event})
    return {"conversation_id": "vi_fd_000001", "topic": "lịch hẹn", "style": "tự nhiên", "rounds": rounds}


class SchemaTest(unittest.TestCase):
    def test_valid(self) -> None:
        self.assertEqual(validate_conversation(valid_row()), [])

    def test_backchannel_cannot_have_reply(self) -> None:
        row = valid_row()
        row["rounds"][2]["event"]["agent_after"] = "Tôi sẽ trả lời cue này."
        self.assertIn("round_2:backchannel_has_reply", validate_conversation(row))

    def test_interruption_requires_reply(self) -> None:
        row = valid_row()
        row["rounds"][4]["event"]["agent_after"] = ""
        self.assertIn("round_4:interruption_missing_reply", validate_conversation(row))


if __name__ == "__main__":
    unittest.main()
