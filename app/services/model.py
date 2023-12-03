import tempfile
import subprocess
import ffmpeg
import os
import openai
import math

from app.util.logger import get_logger

from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate

from app.models.summary import SimpleSummary

import tiktoken

logger = get_logger(__name__)


class Transcriber:
    @staticmethod
    def extract_audio_from_video(
        input_tempfile: tempfile.NamedTemporaryFile,
    ) -> tempfile.NamedTemporaryFile:
        # 音声を抽出するための一時ファイルを作成
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp3"
        ) as output_tempfile:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    input_tempfile.name,
                    "-q:a",
                    "0",
                    "-map",
                    "a",
                    output_tempfile.name,
                ]
            )
        os.remove(input_tempfile.name)
        return output_tempfile

    @classmethod
    def compress_audio(
        cls,
        input_file: tempfile.NamedTemporaryFile,
    ) -> tempfile.NamedTemporaryFile:
        logger.info("=== compress audio ===")

        # Check audio duration
        duration = cls.get_audio_duration(input_file.name)

        # Calculate bitrate based on audio duration
        bitrate = cls.calculate_bitrate(duration)
        logger.info(f"Target bitrate: {bitrate}")

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp3"
        ) as output_tempfile:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    input_file.name,
                    "-codec:a",
                    "mp3",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-b:a",
                    bitrate,
                    output_tempfile.name,
                ]
            )

        os.remove(input_file.name)
        logger.info(f"Compressed audio size:{os.path.getsize(output_tempfile.name)}")
        return output_tempfile

    @staticmethod
    def calculate_bitrate(duration: float) -> str:
        TARGET_FILE_SIZE = 25000000
        target_kbps = int(math.floor(TARGET_FILE_SIZE * 8 / duration / 1000 * 0.9))
        return f"{target_kbps}k"

    @staticmethod
    def is_video_file(filename: str) -> bool:
        video_extensions = [".mp4"]  # TODO 他の拡張子のファイルのサポート
        _, ext = os.path.splitext(filename)
        return ext.lower() in video_extensions

    @staticmethod
    def is_acceptable_file(filename: str) -> bool:
        video_extensions = [".mp4", ".mp3", ".wav", ".m4a"]  # TODO 他の拡張子のファイルのサポート
        _, ext = os.path.splitext(filename)
        return ext.lower() in video_extensions

    @staticmethod
    def get_audio_duration(filepath) -> float:
        try:
            probe = ffmpeg.probe(filepath)
            duration = float(probe["format"]["duration"])
            logger.info(f"Audio duration: {duration}")
        except ffmpeg.Error as e:
            print("Error encountered during ffprobe:", e.stderr.decode())
            raise
        return duration


class MinutesSummarizer:
    # TODO config
    # MODEL = "gpt-3.5-turbo-16k"
    # CHUNK_SIZE = 8000
    # MAX_TOKENS = 15000
    # max_tokensはcompletionのみで計算されているっぽい
    # you requested 19483 tokens (4483 in the messages, 15000 in the completion)
    MODEL = "gpt-4-0613"
    CHUNK_SIZE = 4000
    MAX_TOKENS = 7500

    chunk_overlap = 100
    COST_DICT = {
        "gpt-4-0613": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
    }

    @staticmethod
    def to_numbered_list_str(items):
        return "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))

    @classmethod
    def map_sammaries(cls, text: str):
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator=" ", chunk_size=cls.CHUNK_SIZE, chunk_overlap=cls.chunk_overlap
        )
        text_chunks = text_splitter.split_text(text)
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        costs = 0
        response_messages = []

        for chunk in text_chunks:
            logger.info(chunk)

            messages = [
                {
                    "role": "system",
                    "content": """
                        あなたは会議の議事録を作成するプロフェッショナルアシスタントです。
                        これから会議の文字起こししたテキストを分割して渡します。
                        テキストは話者分離をしていません。
                        この文章から重要な内容を抽出してください。
                        あなたの推察はせず、文章に明記されている内容をそのまま抽出してください。
                        抽出は箇条書きではなく、文章で行なってください。
                        """,
                },
                {"role": "user", "content": chunk},
            ]
            num_tokens = num_tokens_from_messages(messages, model=cls.MODEL)
            response = openai.ChatCompletion.create(
                model=cls.MODEL,
                messages=messages,
                temperature=0,
                max_tokens=min(
                    cls.MAX_TOKENS // len(text_chunks), cls.MAX_TOKENS - num_tokens
                ),
            )
            total_tokens += response["usage"]["total_tokens"]
            prompt_tokens += response["usage"]["prompt_tokens"]
            completion_tokens += response["usage"]["completion_tokens"]
            costs += (
                response["usage"]["prompt_tokens"] * cls.COST_DICT[cls.MODEL]["input"]
                + response["usage"]["completion_tokens"]
                * cls.COST_DICT[cls.MODEL]["output"]
            ) / 1000
            response_messages.append(response["choices"][0]["message"]["content"])
            logger.info(f"chat create: {response['choices'][0]['message']['content']}")
        return response_messages, costs

    @classmethod
    def get_simple_summary(cls, doc_summaries: str):
        costs = 0

        template = """
        あなたは会議の議事録を作成するアシスタントです。
        これから会議の文字起こしの要点を抽出した文章を渡します。
        この会議の要約、要点のリスト、決定事項のリスト、タスクのリストを返してください。
        会議の要約は渡した文章の体裁を留める程度で漏れのない文章で書いてください。タスクのリストには今後やるべきタスクを書いてください。
        タスクのリストには既に完了したことについては記載しないように注意してください。
        決定事項のリストにはタスク以外で決定された事項を書いてください。

        以下は要約のセットである：
        {doc_summaries}"""

        prompt = PromptTemplate(template=template, input_variables=["doc_summaries"])
        messages = [
            {"role": "user", "content": prompt.format(doc_summaries=doc_summaries)},
        ]
        functions = [
            {
                "name": "get_simple_summary",
                "description": """会議の文字起こしの要点などを抽出した文章から会議の要約、会議の要点のリスト、
                    会議で決定された事項のリスト、タスクのリストを抽出するための処理です。""",
                "parameters": SimpleSummary.schema(),
            }
        ]
        message_tokens = num_tokens_from_messages(messages, model=cls.MODEL)
        functions_tokens = num_tokens_from_functions(functions, model=cls.MODEL)
        logger.info(f"message_tokens: {message_tokens}")
        logger.info(f"functions_tokens: {functions_tokens}")
        # TODO response_messagesのtoken数を計算して、token_maxを超えていたら、分割する
        response = openai.ChatCompletion.create(
            model=cls.MODEL,
            messages=messages,
            functions=functions,
            function_call={"name": "get_simple_summary"},
            temperature=0,
            max_tokens=max(
                cls.MAX_TOKENS - message_tokens - functions_tokens, 0
            ),  # tokenをカウントして補正する
        )

        costs += (
            response["usage"]["prompt_tokens"] * cls.COST_DICT[cls.MODEL]["input"]
            + response["usage"]["completion_tokens"]
            * cls.COST_DICT[cls.MODEL]["output"]
        ) / 1000

        return response, costs


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print(
            "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print(
            "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}.
            See https://github.com/openai/openai-python/blob/main/chatml.md
            for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def num_tokens_from_functions(functions, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of functions."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for function in functions:
        function_tokens = len(encoding.encode(function["name"]))
        function_tokens += len(encoding.encode(function["description"]))

        if "parameters" in function:
            parameters = function["parameters"]
            if "properties" in parameters:
                for propertiesKey in parameters["properties"]:
                    function_tokens += len(encoding.encode(propertiesKey))
                    v = parameters["properties"][propertiesKey]
                    for field in v:
                        if field == "type":
                            function_tokens += 2
                            function_tokens += len(encoding.encode(v["type"]))
                        elif field == "description":
                            function_tokens += 2
                            function_tokens += len(encoding.encode(v["description"]))
                        elif field == "enum":
                            function_tokens -= 3
                            for o in v["enum"]:
                                function_tokens += 3
                                function_tokens += len(encoding.encode(o))
                        else:
                            print(f"Warning: not supported field {field}")
                function_tokens += 11

        num_tokens += function_tokens

    num_tokens += 12
    return num_tokens
