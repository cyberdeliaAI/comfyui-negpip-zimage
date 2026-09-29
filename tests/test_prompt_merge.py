from __future__ import annotations

import unittest

from src.prompt_merge import merge_prompts


class PromptMergeTests(unittest.TestCase):
    def test_negative_tags_receive_negative_weights(self):
        self.assertEqual(
            merge_prompts("sharp portrait", "blurry, (text:1.3)"),
            "sharp portrait, (blurry:-1), (text:-1.3)",
        )

    def test_existing_positive_weight_is_preserved(self):
        self.assertEqual(merge_prompts("(detail:1.2)", "noise"), "(detail:1.2), (noise:-1)")

    def test_emoticons_are_preserved_in_positive_prompt(self):
        self.assertEqual(merge_prompts(":3, :d", ""), ":3, :d")

    def test_emoticons_receive_negative_weights_without_losing_text(self):
        self.assertEqual(merge_prompts("portrait", ":3, :d"), "portrait, (:3, :d:-1)")

    def test_unparenthesized_numeric_colons_are_literal(self):
        for prompt in (":3", "16:9", "12:30", "port:8000", "detail:1.3"):
            with self.subTest(prompt=prompt):
                self.assertEqual(merge_prompts(prompt, ""), prompt)

    def test_valid_weights_and_emoticons_can_be_combined(self):
        self.assertEqual(
            merge_prompts(":3, (detail:1.3), :d", "noise"),
            ":3, (detail:1.3), :d, (noise:-1)",
        )

    def test_emoticon_inside_weighted_group_is_preserved(self):
        self.assertEqual(
            merge_prompts("(smile :3, face:1.2)", ""),
            "(smile :3, face:1.2)",
        )

    def test_numeric_colon_inside_group_must_end_at_closing_parenthesis(self):
        for prompt in ("(meeting at 12:30 today)", "(ratio 16:9 composition)"):
            with self.subTest(prompt=prompt):
                self.assertEqual(merge_prompts(prompt, ""), prompt)

    def test_invalid_weight_suffixes_remain_literal(self):
        for prompt in ("(detail:1.2.3)", "(detail:--2)", "(detail:1 2)", "(detail:2x)"):
            with self.subTest(prompt=prompt):
                self.assertEqual(merge_prompts(prompt, ""), prompt)

    def test_escaped_colon_remains_literal(self):
        self.assertEqual(merge_prompts(r"(detail\:1.3)", ""), r"(detail\:1.3)")

    def test_weight_suffix_accepts_signs_decimals_and_surrounding_spaces(self):
        cases = (("+1.3", "1.3"), ("-.5", "-0.5"), ("2.", "2"), (" 0.4 ", "0.4"))
        for suffix, expected in cases:
            with self.subTest(suffix=suffix):
                self.assertEqual(merge_prompts(f"(detail:{suffix})", ""), f"(detail:{expected})")

    def test_nested_weights_still_multiply(self):
        self.assertEqual(merge_prompts("((detail:1.2):1.5)", ""), "(detail:1.8)")

    def test_grouped_negative_weights_still_apply_to_all_tags(self):
        self.assertEqual(
            merge_prompts("", "(blurry, background blur, bokeh:0.4)"),
            "(blurry, background blur, bokeh:-0.4)",
        )


if __name__ == "__main__":
    unittest.main()
