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
                            "description": "Query string to use for full text search (can be empty)",
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
                Specify the exact URL or package name from past messages if they are relevant 
                to the most recent user's message.
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
                    1. If the user wants to talk to a salesperson to buy something, call the 'handover_to_cx' function
                    2. If the user wants something that you cannot provide, call the 'handover_to_cx' function
                    3. If the user is ready to purchase a package/service
                Caution : There is a nuance when the user says "I want..."/"Im looking for..". 
                Based on the chat history, if the user says they want to purchase the package then call this function. 
                If they are simply curious and say something like "I want/looking for a health checkup/treatment", 
                DO NOT call this as its still too general and you can still gather more information.
                """,
                "parameters": {},
            },
        }
    ]


def extract_info_gathered(chat_completion: ChatCompletion):
    response_message = chat_completion.choices[0].message
    package_name = None
    location = None
    budget = None

    if response_message.tool_calls:
        for tool in response_message.tool_calls:
            if tool.type != "function":
                continue
            function = tool.function
            if function.name == "check_info_gathered":
                arg = json.loads(function.arguments)
                package_name = arg.get("package_name")
                location = arg.get("location")
                budget = arg.get("budget")

    return package_name, location, budget


def build_handover_to_bk_function() -> list[ChatCompletionToolParam]:
    return [
        {
            "type": "function",
            "function": {
                "name": "handover_to_bk",
                "description": """
                This function is used to seamlessly transfer the current conversation to the booking team
                when the user's message indicates strongly any mention about RESERVATIONS or POST-PAYMENT
                enquiry for packages.
                """,
                "parameters": {},
            },
        }
    ]


def build_pharmacy_function() -> list[ChatCompletionToolParam]:
    return [
        {
            "type": "function",
            "function": {
                "name": "pharmacy",
                "description": """
                This function is triggered when the user has a strong intent to enquire about 
                pharmacy or medicine related queries.
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
                This function is triggered :
                IF at least 2/3 of the following information has been collected:
                    1. Product/Package Category of user
                    2. Prefered location
                    3. Budget (if any)
                """,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "package_name": {
                            "type": "string",
                            "description": """
                            The name of package/ailment that the customer is looking for.
                            """,
                        },
                        "location": {
                            "type": "string",
                            "description": """
                            The location that the customer would prefer.
                            """,
                        },
                        "budget": {
                            "type": "string",
                            "description": """
                            Price range that the user might be looking for (if applicable).
                            """,
                        },
                    },
                    "required": ["package_name, location"],
                },
            },
        }
    ]


def build_payment_promo_function() -> list[ChatCompletionToolParam]:
    return [
        {
            "type": "function",
            "function": {
                "name": "payment_promo",
                "description": """
                This function is triggered when the user is asking about any promotions/deals in 
                payment methods like credit cards.
                """,
                "parameters": {},
            },
        }
    ]


def build_coupon_function() -> list[ChatCompletionToolParam]:
    return [
        {
            "type": "function",
            "function": {
                "name": "coupon",
                "description": """
                This function is only triggered when the user asks anything related to coupons.
                Like how to claim them or where they can find them
                """,
                "parameters": {},
            },
        }
    ]


def is_payment_promo(chat_completion: ChatCompletion):
    response_message = chat_completion.choices[0].message
    if response_message.tool_calls:
        for tool in response_message.tool_calls:
            if tool.type == "function" and tool.function.name == "payment_promo":
                return True
    return False


def is_coupon(chat_completion: ChatCompletion):
    response_message = chat_completion.choices[0].message
    if response_message.tool_calls:
        for tool in response_message.tool_calls:
            if tool.type == "function" and tool.function.name == "coupon":
                return True
    return False


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


def is_pharmacy(chat_completion: ChatCompletion):
    response_message = chat_completion.choices[0].message
    if response_message.tool_calls:
        for tool in response_message.tool_calls:
            if tool.type == "function" and tool.function.name == "pharmacy":
                return True
    return False
