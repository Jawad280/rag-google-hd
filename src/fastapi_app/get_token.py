import asyncio

from azure.identity import DefaultAzureCredential


async def get_token():
    credential = DefaultAzureCredential(logging_enable=True)
    token = credential.get_token("https://ossrdbms-aad.database.windows.net/.default")
    return token.token


token = asyncio.run(get_token())
print(token)
