import re


def update_urls_with_utm(content: str, pattern: str, utm_source: str = "ai-chat") -> str:
    urls = re.findall(pattern, content)
    updated_urls = [add_utm_param(url, utm_source) for url in urls]

    for old_url, new_url in zip(urls, updated_urls):
        content = content.replace(old_url, new_url)

    return content


def add_utm_param(url: str, utm_source: str = "ai-chat") -> str:
    url = url.rstrip(".")
    if "?" in url:
        return f"{url}&utm_source={utm_source}"
    else:
        return f"{url}?utm_source={utm_source}"


def remove_markdown_elements(content: str) -> str:
    cleaned_content = re.sub(r"\*\*(.*?)\*\*", r"\1", content)
    cleaned_content = re.sub(r"\*(.*?)\*", r"\1", cleaned_content)

    return cleaned_content
