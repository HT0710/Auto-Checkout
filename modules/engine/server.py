import redis


class Server:
    API = redis.Redis(host="localhost", port=6379, decode_responses=True)

    @classmethod
    def set(cls, key: str, value: str):
        cls.API.set(key, value)

    @classmethod
    def get(cls, key: str):
        return cls.API.get(key)
