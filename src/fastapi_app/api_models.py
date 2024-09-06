from typing import Any

from pydantic import BaseModel


class TextContent(BaseModel):
    type: str
    text: str


class ImageUrl(BaseModel):
    url: str
    detail: str = "auto"


class ImageContent(BaseModel):
    type: str
    image_url: ImageUrl


class Message(BaseModel):
    role: str = "user"
    content: str | list[TextContent | ImageContent]


class ChatRequest(BaseModel):
    messages: list[Message]
    context: dict = {}


class ThoughtStep(BaseModel):
    title: str
    description: Any
    props: dict = {}
