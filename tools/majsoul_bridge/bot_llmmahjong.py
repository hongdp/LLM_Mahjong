""" MahjongCopilot bot plugin: LLM_Mahjong DNN agent over local HTTP.

Drop this file into <MahjongCopilot>/bot/llmmahjong/bot_llmmahjong.py
(tools/majsoul_bridge/install.py does it and registers the bot in
bot/factory.py + common/settings.py). The agent itself runs in the
LLM_Mahjong repo:  PYTHONPATH=. python scripts/serve_mjai_bot.py --ckpt ...
"""
import time

import requests

from common.log_helper import LOGGER
from common.mj_helper import MjaiType
from bot.bot import Bot, GameMode


class BotLlmMahjong(Bot):
    """ MJAI bot backed by LLM_Mahjong's scripts/serve_mjai_bot.py """
    retries = 3
    retry_interval = 0.3

    def __init__(self, url: str = "http://127.0.0.1:8765", timeout: float = 5.0) -> None:
        super().__init__("LLM_Mahjong DNN")
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.ignore_next_turn_self_reach: bool = False
        info = self._get("/health")
        self.ckpt = info.get("ckpt", "?")
        LOGGER.info("LLM_Mahjong bot server OK: %s", info)

    @property
    def supported_modes(self) -> list[GameMode]:
        return [GameMode.MJ4P]

    @property
    def info_str(self) -> str:
        return f"{self.name} [{self.ckpt.split('/')[-1]}] @ {self.url}"

    # ---- http ----
    def _get(self, path: str) -> dict:
        r = requests.get(self.url + path, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, payload: dict) -> dict:
        err = None
        for _ in range(self.retries):
            try:
                r = requests.post(self.url + path, json=payload, timeout=self.timeout)
                r.raise_for_status()
                return r.json()
            except Exception as e:      # noqa: BLE001
                err = e
                time.sleep(self.retry_interval)
        raise err

    # ---- Bot interface ----
    def _init_bot_impl(self, mode: GameMode = GameMode.MJ4P):
        self._post("/start", {"seat": self.seat})
        self.ignore_next_turn_self_reach = False

    def _drop_dup_reach(self, msg: dict) -> bool:
        """ The server answers `reach` with `reach_dahai` attached, exactly
        like BotMjai; MahjongCopilot then echoes the reach event back. The
        server-side shadow table must still SEE that reach event (it sets
        the riichi flag), so unlike BotMjai we never drop it. """
        return False

    def react(self, input_msg: dict) -> dict | None:
        res = self._post("/react", {"msg": input_msg})
        return res.get("reaction")

    def react_batch(self, input_list: list[dict]) -> dict | None:
        if not input_list:
            return None
        res = self._post("/react_batch", {"msgs": input_list})
        return res.get("reaction")
