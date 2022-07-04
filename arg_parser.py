import re
from ast import literal_eval
from re import Pattern
from typing import AnyStr, Any

from config import FullConfig

REPLACER_CLASS_COUNT = "??classes"


class Parser:

    __tuple_regex: Pattern[AnyStr]
    __replacer_map: dict[str, Any]

    def __init__(self, cfg: FullConfig, num_classes: int):
        # self.__tuple_regex = re.compile(
        #    "^\([\s]*[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)[\s]*,[\s]*[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)[\s]*\)$")
        self.__tuple_regex = re.compile(
            "^\(.*\)$"
        )
        self.__replacer_map = {
            REPLACER_CLASS_COUNT: num_classes,
        }

    def parse_arg(self, arg: str):
        if type(arg) == str:
            if arg in self.__replacer_map:
                return self.__replacer_map[arg]
            elif self.__tuple_regex.match(arg):
                return literal_eval(arg)
        return arg

    def parse_args(self, args: dict) -> dict[str, any]:
        parsed = {}
        for key, value in args.items():
            parsed[key] = self.parse_arg(value)
        return parsed
