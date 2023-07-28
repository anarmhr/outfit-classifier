from enum import  Enum
from json import JSONEncoder


class Status(Enum):
    OK = 'ok',
    FAIL = 'fail'


class ServiceResponse:
    def __init__(self, status, message, data):
        self.status = status
        self.message = message
        self.data = data

    def serialize(self):
        return {'status': self.status,
                'message': self.message,
                'data': self.data}


class ClassificationResponse:
    def __init__(self, category, colors):
        self.category = category
        self.colors = colors

    def serialize(self):
        return {'category': self.category,
                'colors': str(self.colors)}