import json

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionToolParam,
)


def build_search_function() -> list[ChatCompletionToolParam]:
    return [
        {
            "type": "function",
            "function": {
                "name": "search_database",
                "description": "Search PostgreSQL database for relevant products based on user query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_query": {
                            "type": "string",
                            "description": "Query string to use for full text search, e.g. 'ตรวจสุขภาพ'",
                        },
                        "price_filter": {
                            "type": "object",
                            "description": "Filter search results based on price in Thai Baht of the package",
                            "properties": {
                                "comparison_operator": {
                                    "type": "string",
                                    "description": "Operator to compare the column value, either '>', '<', '>=', '<=', '='",  # noqa
                                },
                                "value": {
                                    "type": "number",
                                    "description": "Value to compare against, e.g. 30",
                                },
                            },
                        },
                        "url_filter": {
                            "type": "object",
                            "description": "Filter search results based on url of the package. The url is package specific.",
                            "properties": {
                                "comparison_operator": {
                                    "type": "string",
                                    "description": "Operator to compare the column value, either '=' or '!='",
                                },
                                "value": {
                                    "type": "string",
                                    "description": """
                                    The package URL to compare against.
                                    Don't pass anything if you can't specify the exact URL from user query.
                                    """,
                                },
                            },
                        },
                    },
                    "required": ["search_query", "url_filter"],
                },
            },
        }
    ]


def extract_search_arguments(chat_completion: ChatCompletion):
    response_message = chat_completion.choices[0].message
    search_query = None
    filters = []
    if response_message.tool_calls:
        for tool in response_message.tool_calls:
            if tool.type != "function":
                continue
            function = tool.function
            if function.name == "search_database":
                arg = json.loads(function.arguments)
                print(arg)
                search_query = arg.get("search_query")
                if "price_filter" in arg and arg["price_filter"]:
                    price_filter = arg["price_filter"]
                    filters.append(
                        {
                            "column": "price",
                            "comparison_operator": price_filter["comparison_operator"],
                            "value": price_filter["value"],
                        }
                    )
                if "url_filter" in arg and arg["url_filter"]:
                    url_filter = arg["url_filter"]
                    if url_filter["value"] != "https://hdmall.co.th":
                        filters.append(
                            {
                                "column": "url",
                                "comparison_operator": url_filter["comparison_operator"],
                                "value": url_filter["value"],
                            }
                        )
    elif query_text := response_message.content:
        search_query = query_text.strip()
    return search_query, filters
