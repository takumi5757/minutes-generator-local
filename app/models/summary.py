from pydantic import BaseModel, Field
from typing import List


class Agendum(BaseModel):
    agenda_titles: List[str] = Field(..., description="アジェンダのタイトルのリスト")


class AgendaItem(BaseModel):
    agenda_title: str = Field(..., description="アジェンダのタイトル")
    agenda_summary: str = Field(..., description="そのアジェンダの要約の内容")


class AgendaItemBullet(BaseModel):
    summary_bullet: List[str] = Field(..., description="要点のリスト")
    decisions: List[str] = Field(..., description="決定事項のリスト")
    next_actions: List[str] = Field(..., description="次回のアクションのリスト")
    questions: List[str] = Field(..., description="質疑応答内容のリスト")


class Summary(BaseModel):
    summary: str = Field(..., description="要約の内容")
    agenda_items: list[AgendaItem] = Field(..., description="各議題毎の会議内容のリスト")


class SimpleSummary(BaseModel):
    summary: str = Field(..., description="要約の内容を文章で書いたもの")
    summary_bullet: List[str] = Field(..., description="要点のリスト")
    decisions: List[str] = Field(..., description="タスク以外で決定された事項リスト")
    tasks: List[str] = Field(..., description="やるべきタスクのリスト")
