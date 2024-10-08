import copy
import json
import logging
import pathlib
from collections.abc import AsyncGenerator
from typing import Any

import requests
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai_messages_token_helper import get_token_limit
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_random_exponential

from .api_models import ThoughtStep
from .llm_tools import (
    build_check_info_gathered_function,
    build_clear_history_function,
    build_coupon_function,
    build_generic_query_function,
    build_google_search_function,
    build_handover_to_bk_function,
    build_handover_to_cx_function,
    build_immediate_handover_function,
    build_installements_query_function,
    build_payment_promo_function,
    build_payment_query_function,
    build_pharmacy_function,
    build_specify_package_function,
    build_welcome_intent_function,
    extract_info_gathered,
    extract_package_name,
    extract_search_arguments,
    extract_url,
    handle_specify_package_function_call,
    is_clear_history,
    is_coupon,
    is_gathered_info,
    is_generic_query,
    is_handover_to_bk,
    is_handover_to_cx,
    is_immediate_handover,
    is_installments_query,
    is_payment_promo,
    is_payment_query,
    is_pharmacy,
    is_welcome_intent,
)
from .postgres_searcher import PostgresSearcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedRAGChat:
    def __init__(
        self,
        *,
        searcher: PostgresSearcher,
        openai_chat_client: AsyncOpenAI,
        chat_model: str,
        chat_deployment: str | None,  # Not needed for non-Azure OpenAI
    ):
        self.searcher = searcher
        self.openai_chat_client = openai_chat_client
        self.chat_model = chat_model
        self.chat_deployment = chat_deployment
        self.chat_token_limit = get_token_limit(chat_model, default_to_minimum=True)
        current_dir = pathlib.Path(__file__).parent
        self.specify_package_prompt_template = open(current_dir / "prompts/specify_package.txt").read()
        self.query_prompt_template = open(current_dir / "prompts/query.txt").read()
        self.answer_prompt_template = open(current_dir / "prompts/answer.txt").read()
        self.interpret_prompt_template = open(current_dir / "prompts/interpret.txt").read()
        self.gather_template = open(current_dir / "prompts/gather.txt").read()
        self.credit_card = open(current_dir / "prompts/credit_card.txt").read()
        self.coupon_template = open(current_dir / "prompts/coupon.txt").read()
        self.promo_template = open(current_dir / "prompts/promo.txt").read()
        self.pharmacy_template = open(current_dir / "prompts/pharmacy.txt").read()
        self.payment_template = open(current_dir / "prompts/payment.txt").read()
        self.installment_template = open(current_dir / "prompts/installment.txt").read()

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def openai_chat_completion(self, *args, **kwargs) -> ChatCompletion:
        return await self.openai_chat_client.chat.completions.create(*args, **kwargs)

    @retry(
        wait=wait_random_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def get_payment_promos(self):
        url = "https://script.google.com/macros/s/AKfycbw18wXh1o6xiD2WY3wcvkQXGZNn4AY2loJjdEqfBGC22xtluoz27L7VeiAyrcMRsFf6fw/exec"

        try:
            body = {"info": "credit_card", "highlight_name": "", "highlight_url": "", "package_url": ""}
            res = requests.post(url=url, json=body)
            res.raise_for_status()
            data = res.json()

            payment_promos = "\n".join(
                f"""
                    promoName: {promo.get('promoName')}\n
                    type: {promo.get('type', '')}\n
                    keyBenefit: {promo.get('keyBenefit', '')}\n
                    url: {promo.get('url', '')}\n
                """
                for promo in data
            )

            return payment_promos
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return ""

    @retry(
        wait=wait_random_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def get_highlight_info(self, highlight_name, highlight_url):
        url = "https://script.google.com/macros/s/AKfycbw18wXh1o6xiD2WY3wcvkQXGZNn4AY2loJjdEqfBGC22xtluoz27L7VeiAyrcMRsFf6fw/exec"

        try:
            body = {
                "info": "highlight",
                "highlight_name": highlight_name,
                "highlight_url": highlight_url,
                "package_url": "",
            }
            res = requests.post(url=url, json=body)
            res.raise_for_status()
            return res.json()
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return ""

    @retry(
        wait=wait_random_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def get_highlight_tags(self):
        url = "https://script.google.com/macros/s/AKfycbw18wXh1o6xiD2WY3wcvkQXGZNn4AY2loJjdEqfBGC22xtluoz27L7VeiAyrcMRsFf6fw/exec"

        try:
            body = {
                "info": "highlight_tags",
                "highlight_name": "",
                "highlight_url": "",
                "package_url": "",
            }
            res = requests.post(url=url, json=body)
            res.raise_for_status()
            data = res.json()
            highlight_tags = data.get("highlightTags")
            highlight_tags = "\n".join(tag for tag in highlight_tags)
            return highlight_tags
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return ""

    @retry(
        wait=wait_random_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def get_payment_method(self, package_url: str):
        url = "https://script.google.com/macros/s/AKfycbw18wXh1o6xiD2WY3wcvkQXGZNn4AY2loJjdEqfBGC22xtluoz27L7VeiAyrcMRsFf6fw/exec"

        try:
            body = {"info": "payment_method", "highlight_name": "", "highlight_url": "", "package_url": package_url}
            res = requests.post(url=url, json=body)
            res.raise_for_status()
            data = res.json()
            res = data.get("paymentMethod")
            return res
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return ""

    @retry(
        wait=wait_random_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def get_cash_discount(self, package_url: str):
        url = "https://script.google.com/macros/s/AKfycbw18wXh1o6xiD2WY3wcvkQXGZNn4AY2loJjdEqfBGC22xtluoz27L7VeiAyrcMRsFf6fw/exec"

        try:
            body = {"info": "discount", "highlight_name": "", "highlight_url": "", "package_url": package_url}
            res = requests.post(url=url, json=body)
            res.raise_for_status()
            data = res.json()
            if data:
                return data
            else:
                return ""
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return ""

    async def google_search(self, messages):
        # Generate an optimized keyword search query based on the chat history and the last question
        query_messages = copy.deepcopy(messages)

        highlight_tags = self.get_highlight_tags()
        query_messages.insert(0, {"role": "system", "content": self.query_prompt_template})
        query_messages[-1]["content"].append({"type": "text", "text": "\n\TAGS:\n" + highlight_tags})
        query_response_token_limit = 500

        query_chat_completion: ChatCompletion = await self.openai_chat_completion(
            messages=query_messages,
            model=self.chat_deployment if self.chat_deployment else self.chat_model,
            temperature=0.0,
            max_tokens=query_response_token_limit,
            n=1,
            tools=build_google_search_function(),
            tool_choice={"type": "function", "function": {"name": "search_google"}},
        )

        search_query, locations = extract_search_arguments(query_chat_completion)
        locations = [f'"{location}"' for location in locations] if locations else []

        if locations:
            # If locations are present in query -> results are likely to be more wider -> add exactTerm to
            # ensure its still relevant
            query_text = f"{search_query} {' OR '.join(locations)}"
            packages, is_package_found = await self.searcher.google_search(
                query_text=query_text, exact_term=search_query, top=3
            )
        else:
            query_text = search_query
            packages, is_package_found = await self.searcher.google_search(
                query_text=query_text, exact_term=None, top=3
            )

        if is_package_found:
            first_result = packages[0]
            sources_content = [f"[{(package.url)}]:{package.to_str_for_broad_rag()}\n\n" for package in packages]

            filter_url = f'https://hdmall.co.th/search?q={first_result.category.replace(" ", "+")}'

            thought_steps = [
                ThoughtStep(title="Prompt to generate search arguments", description=query_messages, props={}),
                ThoughtStep(title="Google Search query", description=query_text, props={}),
                ThoughtStep(
                    title="Google Search results", description=[result.to_dict() for result in packages], props={}
                ),
                ThoughtStep(title="Url to suggest for the filter search", description=filter_url, props={}),
            ]
        else:
            sources_content = []
            filter_url = "https://hdmall.co.th"
            thought_steps = [
                ThoughtStep(title="Prompt to generate search arguments", description=query_messages, props={}),
                ThoughtStep(title="Google Search query", description=query_text, props={}),
                ThoughtStep(title="Google Search results", description=[result for result in packages], props={}),
                ThoughtStep(title="Url to suggest for the filter search", description=filter_url, props={}),
            ]

        return sources_content, thought_steps, filter_url, search_query

    async def run(self, messages: list[dict]) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        # Normalize the message format
        for message in messages:
            if isinstance(message["content"], str):
                message["content"] = [{"type": "text", "text": message["content"]}]

        thought_steps = []

        # Generate a prompt to specify the package if the user is referring to a specific package
        specify_package_messages = copy.deepcopy(messages)
        specify_package_messages.insert(0, {"role": "system", "content": self.specify_package_prompt_template})
        specify_package_token_limit = 300

        specify_package_chat_completion: ChatCompletion = await self.openai_chat_completion(
            messages=specify_package_messages,
            model=self.chat_deployment if self.chat_deployment else self.chat_model,
            temperature=0.0,
            max_tokens=specify_package_token_limit,
            n=1,
            tools=build_handover_to_cx_function()
            + build_handover_to_bk_function()
            + build_specify_package_function()
            + build_clear_history_function()
            + build_pharmacy_function()
            + build_coupon_function()
            + build_payment_promo_function()
            + build_welcome_intent_function()
            + build_payment_query_function()
            + build_immediate_handover_function()
            + build_installements_query_function()
            + build_generic_query_function(),
        )

        specify_package_resp = specify_package_chat_completion.model_dump()
        filter_url = None
        cash_discount = None
        sources_content = []

        if is_welcome_intent(specify_package_chat_completion):
            # LLM to answer welcome messages
            print("Welcome triggered")
            welcome_messages = copy.deepcopy(messages)
            welcome_messages.insert(0, {"role": "system", "content": self.answer_prompt_template})
            welcome_response_token_limit = 300

            welcome_chat_completion: ChatCompletion = await self.openai_chat_completion(
                messages=welcome_messages,
                model=self.chat_deployment if self.chat_deployment else self.chat_model,
                temperature=0.0,
                max_tokens=welcome_response_token_limit,
                n=1,
                tools=None,
            )

            chat_resp = welcome_chat_completion.model_dump()

            chat_resp["choices"][0]["context"] = {
                "data_points": "",
                "thoughts": thought_steps
                + [
                    ThoughtStep(
                        title="Prompt to generate answer",
                        description=[str(message) for message in messages],
                        props=(
                            {"model": self.chat_model, "deployment": self.chat_deployment}
                            if self.chat_deployment
                            else {"model": self.chat_model}
                        ),
                    ),
                ],
            }
            return chat_resp

        if is_generic_query(specify_package_chat_completion):
            # LLM to answer generic messages
            print("Generic triggered")
            generic_messages = copy.deepcopy(messages)
            generic_messages.insert(0, {"role": "system", "content": self.answer_prompt_template})
            generic_response_token_limit = 300

            generic_chat_completion: ChatCompletion = await self.openai_chat_completion(
                messages=generic_messages,
                model=self.chat_deployment if self.chat_deployment else self.chat_model,
                temperature=0.0,
                max_tokens=generic_response_token_limit,
                n=1,
                tools=None,
            )

            chat_resp = generic_chat_completion.model_dump()

            chat_resp["choices"][0]["context"] = {
                "data_points": "",
                "thoughts": thought_steps
                + [
                    ThoughtStep(
                        title="Prompt to generate answer",
                        description=[str(message) for message in messages],
                        props=(
                            {"model": self.chat_model, "deployment": self.chat_deployment}
                            if self.chat_deployment
                            else {"model": self.chat_model}
                        ),
                    ),
                ],
            }
            return chat_resp

        if is_pharmacy(specify_package_chat_completion):
            # LLM to answer queries about pharmacy
            pharmacy_messages = copy.deepcopy(messages)
            pharmacy_messages.insert(0, {"role": "system", "content": self.pharmacy_template})
            pharmacy_response_token_limit = 300

            pharmacy_chat_completion: ChatCompletion = await self.openai_chat_completion(
                messages=pharmacy_messages,
                model=self.chat_deployment if self.chat_deployment else self.chat_model,
                temperature=0.0,
                max_tokens=pharmacy_response_token_limit,
                n=1,
                tools=None,
            )

            chat_resp = pharmacy_chat_completion.model_dump()

            chat_resp["choices"][0]["context"] = {
                "data_points": "",
                "thoughts": thought_steps
                + [
                    ThoughtStep(
                        title="Prompt to generate answer",
                        description=[str(message) for message in messages],
                        props=(
                            {"model": self.chat_model, "deployment": self.chat_deployment}
                            if self.chat_deployment
                            else {"model": self.chat_model}
                        ),
                    ),
                ],
            }
            return chat_resp

        if is_payment_query(specify_package_chat_completion):
            # LLM to answer queries about payment
            print("Payment Route triggered")
            package_url = extract_url(specify_package_chat_completion)
            print(package_url)
            payment_method = self.get_payment_method(package_url)
            messages.insert(0, {"role": "system", "content": self.payment_template})
            messages[-1]["content"].append({"type": "text", "text": "\n\Payment Method:\n" + payment_method})
            payment_response_token_limit = 300

            payment_chat_completion: ChatCompletion = await self.openai_chat_completion(
                messages=messages,
                model=self.chat_deployment if self.chat_deployment else self.chat_model,
                temperature=0.0,
                max_tokens=payment_response_token_limit,
                n=1,
                tools=None,
            )

            chat_resp = payment_chat_completion.model_dump()

            chat_resp["choices"][0]["context"] = {
                "data_points": "",
                "thoughts": thought_steps
                + [
                    ThoughtStep(
                        title="Prompt to generate answer",
                        description=[str(message) for message in messages],
                        props=(
                            {"model": self.chat_model, "deployment": self.chat_deployment}
                            if self.chat_deployment
                            else {"model": self.chat_model}
                        ),
                    ),
                ],
            }
            return chat_resp

        if is_payment_promo(specify_package_chat_completion):
            # LLM to answer queries about payment promotions
            print("Payment Promotions route triggered")
            promo_messages = copy.deepcopy(messages)
            payment_promos = self.get_payment_promos()
            promo_messages.insert(0, {"role": "system", "content": self.promo_template})
            payment_promos = "\n".join(payment_promos)
            promo_messages[-1]["content"].append(
                {"type": "text", "text": "\n\nPayment Promotion Sources:\n" + payment_promos}
            )
            promo_response_token_limit = 4096

            promo_chat_completion: ChatCompletion = await self.openai_chat_completion(
                messages=promo_messages,
                model=self.chat_deployment if self.chat_deployment else self.chat_model,
                temperature=0.0,
                max_tokens=promo_response_token_limit,
                n=1,
                tools=None,
            )

            chat_resp = promo_chat_completion.model_dump()

            chat_resp["choices"][0]["context"] = {"data_points": "", "thoughts": thought_steps}
            return chat_resp

        if is_installments_query(specify_package_chat_completion):
            # LLM to answer queries about installments
            print("Installment route triggered")
            installment_messages = copy.deepcopy(messages)
            installment_messages.insert(0, {"role": "system", "content": self.installment_template})
            installment_response_token_limit = 400

            installment_chat_completion: ChatCompletion = await self.openai_chat_completion(
                messages=installment_messages,
                model=self.chat_deployment if self.chat_deployment else self.chat_model,
                temperature=0.0,
                max_tokens=installment_response_token_limit,
                n=1,
                tools=None,
            )

            chat_resp = installment_chat_completion.model_dump()

            chat_resp["choices"][0]["context"] = {"data_points": "", "thoughts": thought_steps}
            return chat_resp

        if is_coupon(specify_package_chat_completion):
            coupon_messages = copy.deepcopy(messages)
            coupon_messages.insert(0, {"role": "system", "content": self.coupon_template})
            coupon_response_token_limit = 300

            coupon_chat_completion: ChatCompletion = await self.openai_chat_completion(
                messages=coupon_messages,
                model=self.chat_deployment if self.chat_deployment else self.chat_model,
                temperature=0.0,
                max_tokens=coupon_response_token_limit,
                n=1,
                tools=None,
            )

            chat_resp = coupon_chat_completion.model_dump()

            chat_resp["choices"][0]["context"] = {
                "data_points": "",
                "thoughts": thought_steps
                + [
                    ThoughtStep(
                        title="Prompt to generate answer",
                        description=[str(message) for message in messages],
                        props=(
                            {"model": self.chat_model, "deployment": self.chat_deployment}
                            if self.chat_deployment
                            else {"model": self.chat_model}
                        ),
                    ),
                ],
            }
            return chat_resp

        if is_clear_history(specify_package_chat_completion):
            specify_package_resp["choices"][0]["message"]["content"] = "QISCUS_CLEAR_HISTORY"
            return specify_package_resp

        if is_handover_to_cx(specify_package_chat_completion):
            # LLM to check if we have gathered the information
            logger.info("Information gathering route...")
            info_messages = copy.deepcopy(messages)
            info_messages.insert(0, {"role": "system", "content": self.gather_template})
            info_response_token_limit = 300

            info_chat_completion: ChatCompletion = await self.openai_chat_completion(
                messages=info_messages,
                model=self.chat_deployment if self.chat_deployment else self.chat_model,
                temperature=0.0,
                max_tokens=info_response_token_limit,
                n=1,
                tools=build_check_info_gathered_function(),
            )

            if is_gathered_info(info_chat_completion):
                # We need to extract the package_name, location, budget
                package_name, location, budget = extract_info_gathered(info_chat_completion)

                # Send the following text
                note_to_be_added = f"Package: {package_name} \nLocation: {location} \nBudget: {budget}"
                specify_package_resp["choices"][0]["message"]["content"] = (
                    f"QISCUS_INTEGRATION_TO_CX: {note_to_be_added}"
                )
                print(specify_package_resp["choices"][0]["message"]["content"])
                return specify_package_resp

            info_resp = info_chat_completion.model_dump()
            info_gathered = info_resp["choices"][0]["message"]["content"]
            logger.info(f"Information gathering question : {info_gathered}")
            thought_steps.extend(
                [
                    ThoughtStep(title="Prompt to gather info", description=info_messages, props={}),
                    ThoughtStep(title="Information gathered", description=info_gathered, props={}),
                ]
            )
            messages[-1]["content"].append(
                {"type": "text", "text": "Ask the following question to user: " + info_gathered}
            )

            # Build messages for the final chat completion
            messages.insert(0, {"role": "system", "content": self.gather_template})

            response_token_limit = 4096

            chat_completion_response = await self.openai_chat_completion(
                model=self.chat_deployment if self.chat_deployment else self.chat_model,
                messages=messages,
                temperature=0,
                max_tokens=response_token_limit,
                n=1,
                stream=False,
            )
            chat_resp = chat_completion_response.model_dump()

            chat_resp["choices"][0]["context"] = {
                "data_points": {"text": info_gathered},
                "thoughts": thought_steps
                + [
                    ThoughtStep(
                        title="Prompt to generate answer",
                        description=[str(message) for message in messages],
                        props=(
                            {"model": self.chat_model, "deployment": self.chat_deployment}
                            if self.chat_deployment
                            else {"model": self.chat_model}
                        ),
                    ),
                ],
            }
            return chat_resp

        if is_handover_to_bk(specify_package_chat_completion):
            specify_package_resp["choices"][0]["message"]["content"] = "QISCUS_INTEGRATION_TO_BK"
            return specify_package_resp

        if is_immediate_handover(specify_package_chat_completion):
            package_name = extract_package_name(specify_package_chat_completion)
            specify_package_resp["choices"][0]["message"]["content"] = (
                "QISCUS_INTEGRATION_TO_IMMEDIATE_CX: " + package_name
            )
            print("QISCUS_INTEGRATION_TO_IMMEDIATE_CX: " + package_name)
            return specify_package_resp

        specify_package_filters = handle_specify_package_function_call(specify_package_chat_completion)
        highlight_url = extract_url(specify_package_chat_completion)
        highlight_query = ""

        if specify_package_filters:  # Simple SQL search
            results = await self.searcher.simple_sql_search(filters=specify_package_filters)
            if results:
                filter_url = [
                    f'https://hdmall.co.th/search?q={package.category.replace(" ", "+")}' for package in results
                ]

                sources_content = [f"[{(package.url)}]:{package.to_str_for_narrow_rag()}\n\n" for package in results]
                cash_discount = self.get_cash_discount(package_url=results[0].url)
                thought_steps.extend(
                    [
                        ThoughtStep(
                            title="Prompt to specify package",
                            description=[str(message) for message in specify_package_messages],
                            props={"model": self.chat_model, "deployment": self.chat_deployment}
                            if self.chat_deployment
                            else {"model": self.chat_model},
                        ),
                        ThoughtStep(title="Specified package filters", description=specify_package_filters, props={}),
                        ThoughtStep(
                            title="SQL search results", description=[result.to_dict() for result in results], props={}
                        ),
                        ThoughtStep(
                            title="Url to suggest for the filter search",
                            description=filter_url,
                            props={},
                        ),
                    ]
                )
            else:
                print("Google search triggered as couldnt find any packages")
                # No results found with SQL search, fall back to the google search
                sources_content, additional_thought_steps, filter_url, query = await self.google_search(messages)
                highlight_query = query
                thought_steps.extend(additional_thought_steps)
        else:  # Google search
            print("Google search is triggered by default")
            sources_content, additional_thought_steps, filter_url, query = await self.google_search(messages)
            highlight_query = query
            thought_steps.extend(additional_thought_steps)

        content = "\n".join(sources_content)
        highlight_content = ""
        highlight_name = highlight_query
        print(highlight_url, highlight_name)

        if highlight_name or highlight_url:
            result = self.get_highlight_info(highlight_name=highlight_name, highlight_url=highlight_url)
            if result:
                # Found highlight content
                highlight_content = json.dumps(result, ensure_ascii=False)

        # Build messages for the final chat completion
        messages.insert(0, {"role": "system", "content": self.answer_prompt_template})
        messages[-1]["content"].append(
            {"type": "text", "text": "\n\nHighlight Campaign Sources:\n" + highlight_content}
        )
        messages[-1]["content"].append({"type": "text", "text": "\n\nSources:\n" + content})

        # Append cash discount to final message
        if cash_discount:
            messages[-1]["content"].append(
                {
                    "type": "text",
                    "text": f"""
                            \n\nThe current package the user has inquired about has a cash discount!
                            Include the following in your response as well :
                            หากคุณซื้อแพ็กเกจนี้ด้วยการจ่ายเต็มจำนวนผ่าน PromptPay คุณจะได้รับส่วนลดเพิ่ม {cash_discount} บาท
                        """,
                }
            )

        # Append the URL to the final message
        if filter_url:
            messages[-1]["content"].append(
                {
                    "type": "text",
                    "text": f"""
                            \n\nAdd this url at the end of your response (if url is related to the query):{filter_url}
                        """,
                }
            )

        response_token_limit = 4096

        chat_completion_response = await self.openai_chat_completion(
            model=self.chat_deployment if self.chat_deployment else self.chat_model,
            messages=messages,
            temperature=0,
            max_tokens=response_token_limit,
            n=1,
            stream=False,
        )
        chat_resp = chat_completion_response.model_dump()

        chat_resp["choices"][0]["context"] = {
            "data_points": {"text": sources_content},
            "thoughts": thought_steps
            + [
                ThoughtStep(
                    title="Prompt to generate answer",
                    description=[str(message) for message in messages],
                    props=(
                        {"model": self.chat_model, "deployment": self.chat_deployment}
                        if self.chat_deployment
                        else {"model": self.chat_model}
                    ),
                ),
            ],
        }
        return chat_resp
