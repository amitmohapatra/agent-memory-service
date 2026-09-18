"""An injected http_client inherits the base URL it was not given."""
import httpx, pytest
from universal_memory.transport import Transport

def test_injected_client_without_base_url_gets_one():
    client = httpx.AsyncClient()
    Transport("http://memory.test/", api_key="k", client=client)
    assert str(client.base_url) == "http://memory.test"
    assert client.headers["X-API-Key"] == "k"

def test_an_explicit_base_url_on_an_injected_client_is_respected():
    client = httpx.AsyncClient(base_url="http://chosen.test")
    Transport("http://memory.test", client=client)
    assert str(client.base_url) == "http://chosen.test"
