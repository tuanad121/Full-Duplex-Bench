#!/usr/bin/env python3
"""Offline unit tests for matching, validation, and reviewer separation."""

import unittest

from common import choose_reviewer, validate
from generate_batch import make_jobs


def sample(cid="vi_fd_000001_sa"):
    rounds = []
    for i in range(6):
        event = None
        if i == 2:
            event = {"type": "backchannel", "text": "Ừm.", "agent_after": "", "overlap_fraction": .4}
        if i == 4:
            event = {"type": "real_interruption", "text": "Khoan, đổi sang chiều mai nhé.", "agent_after": "Được, tôi sẽ chuyển sang chiều mai.", "overlap_fraction": .5}
        rounds.append({"user": f"Đây là yêu cầu tự nhiên số {i} của tôi.", "agent": f"Tôi đang trả lời chi tiết cho yêu cầu số {i} của bạn.", "event": event})
    return {"conversation_id": cid, "topic": "sắp lịch", "style": "tự nhiên", "rounds": rounds}


class PipelineTests(unittest.TestCase):
    def test_valid_conversation(self):
        self.assertEqual(validate(sample()), [])

    def test_backchannel_reply_rejected(self):
        row = sample()
        row["rounds"][2]["event"]["agent_after"] = "Không cần phản hồi câu này."
        self.assertIn("round_2:backchannel_reply", validate(row))

    def test_independent_reviewer(self):
        selected = choose_reviewer(["anthropic/claude-opus-4.6"], ["anthropic/claude-sonnet-4.6", "openai/gpt-5"])
        self.assertEqual(selected, "openai/gpt-5")

    def test_exact_hybrid_mix(self):
        cfg = {"seed": 1, "strategy_weights": {"single_author": .75, "role_play": .25},
               "rounds_min": 6, "rounds_max": 10}
        jobs = make_jobs(cfg, 1001, 100)
        self.assertEqual(sum(x["strategy"] == "single_author" for x in jobs), 75)
        self.assertEqual(sum(x["strategy"] == "role_play" for x in jobs), 25)


if __name__ == "__main__":
    unittest.main()
