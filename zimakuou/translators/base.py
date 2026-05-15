from abc import ABC, abstractmethod


class Translator(ABC):
    @abstractmethod
    def translate(self, text: str, context: list[str]) -> str:
        """Translate one Japanese cue to Traditional Chinese.

        `context` is the list of prior Japanese cues — pass a sliding window
        so the LLM keeps pronouns and honorifics coherent across cues.
        """
