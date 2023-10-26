import json
from enum import Enum
from json import JSONEncoder
from dataclasses import dataclass
from flask import make_response


class Status(Enum):
    OK = 'ok',
    FAIL = 'fail'


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "__dict__"):
            return obj.__dict__

        return super().default(obj)


@dataclass
class ServiceResponse:
    def __init__(self, status, message=None, data=None):
        self.status = status.name
        self.message = message

        if data is not None:
            self.data = vars(data)

    def make_response(self):
        response_code = 200 if self.status == Status.OK.name else 500
        return make_response(vars(self), response_code)


@dataclass
class ClassificationResponse:
    def __init__(self, category, colors):
        self.category = category
        self.colors = colors
