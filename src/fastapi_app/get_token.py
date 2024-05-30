import asyncio
import os

from azure.identity import DefaultAzureCredential

async def get_token():
    credential = DefaultAzureCredential()
    token = credential.get_token("https://ossrdbms-aad.database.windows.net/.default")
    return token.token


token = asyncio.run(get_token())
print(token)