import json

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionToolParam,
)


def build_google_search_function() -> list[ChatCompletionToolParam]:
    return [
        {
            "type": "function",
            "function": {
                "name": "search_google",
                "description": "Search for relevant products based on user query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_query": {
                            "type": "string",
                            "description": "Query string to use for full text search, e.g. 'ตรวจสุขภาพ'",
                        },
                        "locations": {
                            "type": "array",
                            "items": {
                                "type": "string",
                            },
                            "description": """
                            Translate all inputs to thai.
                            A list of nearby districts(Amphoes) from what the user provides.
                            For example, if the user says `รังสิต`, the locations should be 
                            [`รังสิต`, `ธัญบุรี`, `เมืองปทุมธานี`, `คลองหลวง`, `ลำลูกกา`]. The location the user provided should
                            be the first in the response and followed by areas surrounding it.
                            Only parse this property if the user specifies an area, not a specific place.
                            """,
                        },
                    },
                    "required": ["search_query"],
                },
            },
        }
    ]


def extract_search_arguments(chat_completion: ChatCompletion):
    response_message = chat_completion.choices[0].message
    search_query = None

    if response_message.tool_calls:
        for tool in response_message.tool_calls:
            if tool.type != "function":
                continue
            function = tool.function
            if function.name == "search_google":
                arg = json.loads(function.arguments)
                search_query = arg.get("search_query")
                locations = arg.get("locations", [])

    return search_query, locations


def build_specify_package_function() -> list[ChatCompletionToolParam]:
    return [
        {
            "type": "function",
            "function": {
                "name": "specify_package",
                "description": """
                Specify the exact URL or package name from past messages if they are relevant to the most recent user's 
                message.
                This tool is intended to find specific packages previously mentioned and should not be used for general 
                inquiries or price-based requests.
                """,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": """
                            The exact URL of the package from past messages,
                            e.g. 'https://hdmall.co.th/dental-clinics/xray-for-orthodontics-1-csdc'
                            If it includes any UTM parameters, please remove them.
                            """,
                        },
                        "package_name": {
                            "type": "string",
                            "description": """
                            The exact package name from past messages,
                            always contains the package name and the hospital name,
                            e.g. 'เอกซเรย์สำหรับการจัดฟัน ที่ CSDC'
                            """,
                        },
                    },
                    "required": [],
                },
            },
        }
    ]


def handle_specify_package_function_call(chat_completion: ChatCompletion):
    response_message = chat_completion.choices[0].message
    filters = []
    if response_message.tool_calls:
        for tool in response_message.tool_calls:
            if tool.type == "function" and tool.function.name == "specify_package":
                args = json.loads(tool.function.arguments)
                url = args.get("url")
                package_name = args.get("package_name")
                if url:
                    filters.append(
                        {
                            "column": "url",
                            "comparison_operator": "ILIKE",
                            "value": f"%{url}%",
                        }
                    )
                if package_name:
                    filters.append(
                        {
                            "column": "package_name",
                            "comparison_operator": "ILIKE",
                            "value": f"%{package_name}%",
                        }
                    )
    return filters


def build_handover_to_cx_function() -> list[ChatCompletionToolParam]:
    return [
        {
            "type": "function",
            "function": {
                "name": "handover_to_cx",
                "description": """
                This function is used to seamlessly transfer the current conversation to a live
                customer support agent/human/someone when the user's message indicates the following :
                1. Any mentions about payment or wanting to make payment. 
                2. Specific HDMall service queries that you are not able to provide further information about.
                3. Based on the interpret llm, if the user is in a clear decision stage and wants more information.
                4. Wants to talk to someone
                """,
                "parameters": {},
            },
        }
    ]


def build_handover_to_bk_function() -> list[ChatCompletionToolParam]:
    return [
        {
            "type": "function",
            "function": {
                "name": "handover_to_bk",
                "description": """
                This function is used to seamlessly transfer the current conversation to the booking team
                when the user's message indicates strongly any mention about reservations or post-payment 
                enquiry for packages.
                """,
                "parameters": {},
            },
        }
    ]


def build_app_link_function() -> list[ChatCompletionToolParam]:
    return [
        {
            "type": "function",
            "function": {
                "name": "app_link",
                "description": """
                This function is used to provide a download link for our app : https://www.example.com. You trigger 
                this function when the user has a strong intent to enquire about pharmacy or medicine related queries.
                All you have to do is simply return the download link in the response.
                """,
                "parameters": {},
            },
        }
    ]


def build_clear_history_function() -> list[ChatCompletionToolParam]:
    return [
        {
            "type": "function",
            "function": {
                "name": "clear_history",
                "description": """
                This function is used to clear all the past chat history between the user and the chatbot. 
                This will be handled in the middleware. To trigger this in the middleware, you simply have
                to return a string 'clear_history'. When the user mentions anything about clearing the chat history,
                this function must be activated.
                """,
                "parameters": {},
            },
        }
    ]


def build_check_info_gathered_function() -> list[ChatCompletionToolParam]:
    return [
        {
            "type": "function",
            "function": {
                "name": "check_info_gathered",
                "description": """
                This function is also triggered : 
                - IF there is any mention about payment or wanting to pay.
                OR
                - IF at least 2/3 of the following information has been collected:
                    1. Product/Package Category of user
                    2. Prefered location
                    3. Budget (if any)
                """,
                "parameters": {},
            },
        }
    ]


def is_gathered_info(chat_completion: ChatCompletion):
    response_message = chat_completion.choices[0].message
    if response_message.tool_calls:
        for tool in response_message.tool_calls:
            if tool.type == "function" and tool.function.name == "check_info_gathered":
                return True
    return False


def is_clear_history(chat_completion: ChatCompletion):
    response_message = chat_completion.choices[0].message
    if response_message.tool_calls:
        for tool in response_message.tool_calls:
            if tool.type == "function" and tool.function.name == "clear_history":
                return True
    return False


def is_handover_to_cx(chat_completion: ChatCompletion):
    response_message = chat_completion.choices[0].message
    if response_message.tool_calls:
        for tool in response_message.tool_calls:
            if tool.type == "function" and tool.function.name == "handover_to_cx":
                return True
    return False


def is_handover_to_bk(chat_completion: ChatCompletion):
    response_message = chat_completion.choices[0].message
    if response_message.tool_calls:
        for tool in response_message.tool_calls:
            if tool.type == "function" and tool.function.name == "handover_to_bk":
                return True
    return False


def is_app_link(chat_completion: ChatCompletion):
    response_message = chat_completion.choices[0].message
    if response_message.tool_calls:
        for tool in response_message.tool_calls:
            if tool.type == "function" and tool.function.name == "app_link":
                return True
    return False
