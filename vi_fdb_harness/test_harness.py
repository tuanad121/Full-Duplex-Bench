import json
from pathlib import Path
import tempfile
import unittest

from harness import load_manifest
from transcribe import normalize_chunkformer, timestamp_seconds
from judge import distractor_content, lost_pause_suffix, missed_interruption, premature_pause_turn


class ChunkFormerNormalizationTest(unittest.TestCase):
    def test_documented_timestamp_text(self):
        raw = "[00:00:01.200] - [00:00:02.400]: xin chào"
        result = normalize_chunkformer(raw)
        self.assertEqual(result["text"], "xin chào")
        self.assertEqual(result["chunks"][0]["timestamp"], [1.2, 2.4])

    def test_dict_segments(self):
        result = normalize_chunkformer({
            "text": "mình nói tiếp",
            "segments": [{"text": "mình nói tiếp", "start": 4.9, "end": 5.8}],
        })
        self.assertEqual(result["text"], "mình nói tiếp")
        self.assertEqual(result["chunks"][0]["timestamp"], [4.9, 5.8])

    def test_chunkformer_runtime_timestamp(self):
        self.assertEqual(timestamp_seconds("00:00:04:480"), 4.48)

    def test_chunkformer_runtime_result(self):
        result = normalize_chunkformer([
            {"decode": "rồi mình nói tiếp nhé", "start": "00:00:04:480", "end": "00:00:05:520"}
        ])
        self.assertEqual(result["text"], "rồi mình nói tiếp nhé")
        self.assertEqual(result["chunks"][0]["timestamp"], [4.48, 5.52])


class HuggingFaceManifestTest(unittest.TestCase):
    def test_recovers_source_suite_version_from_flattened_release(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sample = root / "background_speech" / "000001"
            sample.mkdir(parents=True)
            (sample / "metadata.json").write_text(
                json.dumps({"source_methodology": "English-FDB-v1.5"}),
                encoding="utf-8",
            )
            (root / "manifest.json").write_text(
                json.dumps([{
                    "task": "background_speech",
                    "id": "000001",
                    "metadata": "background_speech/000001/metadata.json",
                }]),
                encoding="utf-8",
            )

            self.assertEqual(load_manifest(root)[0]["version"], "v1.5")


class JudgeTimingTest(unittest.TestCase):
    def test_pause_response_before_resumption_is_premature(self):
        evidence = premature_pause_turn(
            {"event_text": "[PAUSE]"},
            {"events": [
                {"type": "response.created", "time": 3.598, "response_id": "r1"},
                {"type": "input_audio_buffer.speech_started", "time": 5.108},
                {"type": "response.done", "time": 5.109, "response_id": "r1", "status": "cancelled"},
            ]},
        )
        self.assertEqual(evidence["response_created"], 3.598)

    def test_spoken_interruption_is_not_pause_override(self):
        evidence = premature_pause_turn(
            {"event_text": "Khoan, đổi yêu cầu giúp tôi."},
            {"events": [
                {"type": "response.created", "time": 3.0, "response_id": "r1"},
                {"type": "input_audio_buffer.speech_started", "time": 5.0},
                {"type": "response.done", "time": 5.1, "response_id": "r1", "status": "cancelled"},
            ]},
        )
        self.assertIsNone(evidence)

    def test_lost_post_pause_words_is_failure(self):
        evidence = lost_pause_suffix(
            {"event_text": "[PAUSE]", "primary_text": "phòng đôi yên ... tĩnh cho tối mai được không"},
            {"text": "mình sẽ kiểm tra phòng đôi yên cho bạn"},
            {"events": [
                {"type": "response.created", "time": 3.6, "response_id": "r1"},
                {"type": "input_audio_buffer.speech_started", "time": 5.1},
                {"type": "response.done", "time": 5.1, "response_id": "r1", "status": "cancelled"},
            ]},
        )
        self.assertEqual(evidence["suffix_tokens"], ["mai", "tĩnh", "tối"])

    def test_retained_post_pause_words_is_not_forced_failure(self):
        evidence = lost_pause_suffix(
            {"event_text": "[PAUSE]", "primary_text": "thủ tục cho ... chuyến bay đi Đà Nẵng nhé"},
            {"text": "mình sẽ hướng dẫn thủ tục cho chuyến bay đi đà nẵng"},
            {"events": [
                {"type": "response.created", "time": 5.5, "response_id": "r1"},
                {"type": "input_audio_buffer.speech_started", "time": 6.3},
                {"type": "response.done", "time": 6.3, "response_id": "r1", "status": "cancelled"},
            ]},
        )
        self.assertIsNone(evidence)

    def test_changed_request_without_later_response_is_missed(self):
        evidence = missed_interruption(
            {"task": "user_interruption", "timestamps": [5.88, 9.32]},
            {"events": [
                {"type": "response.created", "time": 3.39},
                {"type": "response.done", "time": 5.05},
                {"type": "input_audio_buffer.speech_started", "time": 6.18},
            ]},
        )
        self.assertEqual(evidence["interruption_speech_started"], 6.18)

    def test_changed_request_with_later_response_is_not_missed(self):
        evidence = missed_interruption(
            {"task": "user_interruption", "timestamps": [5.0, 7.0]},
            {"events": [
                {"type": "input_audio_buffer.speech_started", "time": 5.2},
                {"type": "response.created", "time": 7.5},
            ]},
        )
        self.assertIsNone(evidence)

    def test_repeating_background_content_is_engagement(self):
        evidence = distractor_content(
            {"task": "background_speech", "primary_text": "chỉ giúp quầy làm thủ tục chuyến bay", "event_text": "mời khách số hai mươi ba tới quầy hướng dẫn"},
            {"text": "mình thấy quầy hướng dẫn số hai mươi ba ngay phía trước"},
        )
        self.assertEqual(evidence["handling"], "engaged")

    def test_staying_on_primary_topic_ignores_background(self):
        evidence = distractor_content(
            {"task": "background_speech", "primary_text": "kiểm tra phòng đôi yên tĩnh", "event_text": "mời khách số hai mươi ba tới quầy hướng dẫn"},
            {"text": "mình sẽ kiểm tra phòng đôi yên tĩnh cho tối mai"},
        )
        self.assertEqual(evidence["handling"], "ignored")


if __name__ == "__main__":
    unittest.main()
