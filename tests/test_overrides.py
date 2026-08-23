"""KanOverride (diagnostic arena wrapper) — forces tenpai ankan, blocks daiminkan."""
import unittest

from src.agents.dnn.overrides import KanOverride
from src.tasks.mahjong.table import PyMahjongTable
from tests.test_engine import rig


class _Net:
    encoder_variant = "v1"


class TestKanOverride(unittest.TestCase):
    def test_tenpai_ankan_is_forced(self):
        t = PyMahjongTable()
        # 234s 5555s 345m 456m 5p tanki, drew the 4th 5s (the dashboard case)
        rig(t, 0, ['2s', '3s', '4s', '5s', '5s', '5s', '5s',
                   '3m', '4m', '4m', '5m', '5m', '6m', '5p'], drawn='5s')
        t.turn = 0
        legal = t.get_legal_actions(0)
        kan = '<action type="kan" tile="5s" />'
        self.assertIn(kan, legal)
        w = KanOverride(_Net())
        self.assertEqual(w.override(t, 0, legal, '<action type="discard" tile="5p" />'), kan)
        self.assertEqual(w.stats["ankan_forced"], 1)
        self.assertEqual(w.encoder_variant, "v1")          # attribute proxy

    def test_non_tenpai_ankan_left_to_net(self):
        t = PyMahjongTable()
        rig(t, 0, ['1m', '1m', '1m', '1m', '4m', '7m', '2p', '5p', '8p',
                   '3s', '6s', '9s', '1z', '2z'], drawn='1m')
        t.turn = 0
        legal = t.get_legal_actions(0)
        self.assertIn('<action type="kan" tile="1m" />', legal)
        d = '<action type="discard" tile="2z" />'
        self.assertEqual(KanOverride(_Net()).override(t, 0, legal, d), d)

    def test_daiminkan_is_blocked_to_skip(self):
        t = PyMahjongTable()
        legal = ['<action type="skip" />', '<action type="pon" tile="3p" />',
                 '<action type="kan" tile="3p" />']
        t.turn = 1                                  # someone else's discard
        w = KanOverride(_Net())
        self.assertEqual(w.override(t, 0, legal, '<action type="kan" tile="3p" />'),
                         '<action type="skip" />')
        self.assertEqual(w.override(t, 0, legal, '<action type="pon" tile="3p" />'),
                         '<action type="pon" tile="3p" />')
        self.assertEqual(KanOverride(_Net(), no_daiminkan=False).override(
            t, 0, legal, '<action type="kan" tile="3p" />'), '<action type="kan" tile="3p" />')


if __name__ == "__main__":
    unittest.main()
