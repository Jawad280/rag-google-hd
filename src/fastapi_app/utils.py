import re


def update_urls_with_utm(content: str, pattern: str, utm_source: str = "ai-chat") -> str:
    urls = re.findall(pattern, content)
    updated_urls = [add_utm_param(url, utm_source) for url in urls]

    for old_url, new_url in zip(urls, updated_urls):
        # Use re.sub to ensure we replace the exact URL without altering similar URLs
        content = re.sub(rf"(?<![^\s]){re.escape(old_url)}(?![^\s])", new_url, content)
    return content


def add_utm_param(url: str, utm_source: str = "ai-chat") -> str:
    url = url.rstrip(".")
    if "utm_source=" in url:
        return url
    if "?" in url:
        return f"{url}&utm_source={utm_source}"
    else:
        return f"{url}?utm_source={utm_source}"


def remove_markdown_elements(content: str) -> str:
    # Remove code blocks (```...```)
    cleaned_content = re.sub(r"```.*?```", "", content, flags=re.DOTALL)

    # Remove inline code (`...`)
    cleaned_content = re.sub(r"`([^`]*)`", r"\1", cleaned_content)

    # Remove headers (e.g., # Header)
    cleaned_content = re.sub(r"^\s*#+\s*(.*)", r"\1", cleaned_content, flags=re.MULTILINE)

    # Remove bold and italic (**text**, *text*, __text__, _text_)
    cleaned_content = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned_content)
    cleaned_content = re.sub(r"\*(.*?)\*", r"\1", cleaned_content)

    return cleaned_content.strip()
