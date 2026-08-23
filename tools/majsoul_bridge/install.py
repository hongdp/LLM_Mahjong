"""Register the LLM_Mahjong bot inside a MahjongCopilot checkout (idempotent).

  python tools/majsoul_bridge/install.py /path/to/MahjongCopilot

Does three things:
  1. copies bot_llmmahjong.py -> <MC>/bot/llmmahjong/
  2. adds a "LLM_Mahjong" case to <MC>/bot/factory.py (+ MODEL_TYPE_STRINGS)
  3. adds `llmmahjong_url` to <MC>/common/settings.py (default http://127.0.0.1:8765)
  4. makes <MC>/game/game_state.py forward kyoku / game results to the bot
     as `end_kyoku` / `end_game` events (MahjongCopilot drops them; we need
     them in the session log to score the agent)
Select model type "LLM_Mahjong" in MahjongCopilot's settings window afterwards
(or set "model_type": "LLM_Mahjong" in its settings.json).
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def patch(path, anchor, insert, marker):
    src = open(path, encoding="utf-8").read()
    if marker in src:
        print(f"  already patched: {path}")
        return
    if anchor not in src:
        sys.exit(f"anchor not found in {path!r}: {anchor!r} — MahjongCopilot layout changed, patch by hand")
    src = src.replace(anchor, anchor + insert, 1)
    open(path, "w", encoding="utf-8").write(src)
    print(f"  patched: {path}")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    mc = os.path.abspath(sys.argv[1])
    for f in ("bot/factory.py", "common/settings.py", "bot/bot.py"):
        if not os.path.exists(os.path.join(mc, f)):
            sys.exit(f"{mc} does not look like a MahjongCopilot checkout (missing {f})")

    dst = os.path.join(mc, "bot", "llmmahjong")
    os.makedirs(dst, exist_ok=True)
    open(os.path.join(dst, "__init__.py"), "a").close()
    shutil.copy(os.path.join(HERE, "bot_llmmahjong.py"), dst)
    print(f"  copied bot -> {dst}")

    patch(os.path.join(mc, "bot", "factory.py"),
          "from .akagiot.bot_akagiot import BotAkagiOt\n",
          "from .llmmahjong.bot_llmmahjong import BotLlmMahjong  # LLM_Mahjong\n",
          "BotLlmMahjong")
    patch(os.path.join(mc, "bot", "factory.py"),
          'MODEL_TYPE_STRINGS = ["Local", "AkagiOT", "MJAPI"',
          ', "LLM_Mahjong"', '"LLM_Mahjong"')
    patch(os.path.join(mc, "bot", "factory.py"),
          '        case "MJAPI":\n            bot = BotMjapi(settings)\n',
          '        case "LLM_Mahjong":\n            bot = BotLlmMahjong(settings.llmmahjong_url)\n',
          'case "LLM_Mahjong"')
    patch(os.path.join(mc, "common", "settings.py"),
          '        self.mjapi_model_select:str = self._get_value("mjapi_model_select","baseline")\n',
          '        self.llmmahjong_url:str = self._get_value("llmmahjong_url", "http://127.0.0.1:8765")  # LLM_Mahjong\n',
          "llmmahjong_url")
    gs = os.path.join(mc, "game", "game_state.py")
    src = open(gs, encoding="utf-8").read()
    if "_llm_forward_end_kyoku" not in src:
        helper = (
            "    def _llm_forward_end_kyoku(self, name, data):\n"
            "        \"\"\" LLM_Mahjong: forward the liqi round result to the bot (for logging)\"\"\"\n"
            "        try:\n"
            "            self.mjai_bot.react({'type': 'end_kyoku', 'liqi_name': name, 'liqi_data': data})\n"
            "        except Exception as e:  # noqa: BLE001\n"
            "            LOGGER.warning('end_kyoku forward failed: %s', e)\n\n"
        )
        anchor = "    def ms_end_kyoku(self) -> dict | None:\n"
        call = "            return self.ms_end_kyoku()\n"
        end_anchor = "            # process end result\n            pass\n"
        end_fwd = (
            "        try:    # LLM_Mahjong: forward final result to the bot (for logging)\n"
            "            self.mjai_bot.react({'type': 'end_game', 'liqi_data': liqi_data})\n"
            "        except Exception as e:  # noqa: BLE001\n"
            "            LOGGER.warning('end_game forward failed: %s', e)\n"
        )
        if anchor not in src or src.count(call) != 3 or end_anchor not in src:
            sys.exit("game_state.py layout changed — apply the result-forwarding patch by hand (see README)")
        src = src.replace(anchor, helper + anchor, 1)
        src = src.replace(call, "            self._llm_forward_end_kyoku(liqi_data_name, liqi_data_data)\n" + call)
        src = src.replace(end_anchor, end_anchor + end_fwd, 1)
        open(gs, "w", encoding="utf-8").write(src)
        print(f"  patched: {gs}")
    else:
        print(f"  already patched: {gs}")
    print("done. Start the agent server, then pick model type 'LLM_Mahjong' in MahjongCopilot settings.")


if __name__ == "__main__":
    main()
