import httpx


client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
