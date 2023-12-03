from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    Query,
    BackgroundTasks,
)
import os
import openai
from typing import Union

from fastapi.responses import JSONResponse
import tempfile
from app.util.logger import get_logger
from app.services.model import Transcriber as tc
from app.services.model import MinutesSummarizer as ms
import json
import time

import requests


import torch
from app.util.split_audio import split_audio_voiced

torch.set_num_threads(1)
logger = get_logger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
slack_token = os.getenv("slack_token")
# Slack Webhook URL
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

api_router = APIRouter()


async def execute_summarize(
    upload_file: UploadFile = File(...),
    prompt: Union[str, None] = Query(default=None),
    response_format: Union[str, None] = Query(
        default="text", enum=["text", "vtt", "srt", "verbose_json", "json"]
    ),
):
    # 一時ファイルの作成
    _, file_extension = os.path.splitext(upload_file.filename)
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=file_extension
    ) as input_tempfile:
        for chunk in upload_file.file:
            input_tempfile.write(chunk)
        input_tempfile.flush()

    logger.info(type(input_tempfile))

    # 有効なファイルかチェック
    if not tc.is_acceptable_file(input_tempfile.name):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # 4時間以上のファイルはエラー
    duration = tc.get_audio_duration(input_tempfile.name)
    if duration > 4 * 60 * 60 + 60:
        raise HTTPException(
            status_code=400, detail="Too long audio file. (max 4 hours)"
        )
    # videoであれば音声を抽出
    if tc.is_video_file(input_tempfile.name):
        logger.info("Extracting audio from video")
        input_tempfile = tc.extract_audio_from_video(input_tempfile)
        logger.info(f"Extracted audio size: {os.path.getsize(input_tempfile.name)}")

    # VADで無音区間を削除
    logger.info("Removing silent parts")
    model, utils = torch.hub.load("/silero-vad/", model="silero_vad", source="local")
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    SAMPLING_RATE = 16000
    wav = read_audio(
        input_tempfile.name,
        sampling_rate=SAMPLING_RATE,
    )
    # get speech timestamps from full audio file
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE)

    split_audio_voiced_data = split_audio_voiced(wav, speech_timestamps)

    transcript = ""
    for segment in split_audio_voiced_data:
        # merge all speech chunks to one audio
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".wav"
        ) as output_tempfile:
            save_audio(
                output_tempfile.name,
                collect_chunks(segment["voiced_segments"], wav),
                sampling_rate=SAMPLING_RATE,
            )

        # whisper APIの25MB制限に対応するために圧縮
        if os.path.getsize(output_tempfile.name) > 25000000:
            compressed_file = tc.compress_audio(output_tempfile)
        else:
            compressed_file = output_tempfile

        # 一時ファイルを使用してtranscribeを実行
        chunk_transcript = openai.Audio.transcribe(
            "whisper-1",
            open(compressed_file.name, "rb"),
            prompt=prompt,
            response_format=response_format,
            temperature=0,
            language="ja",
        )
        transcript += chunk_transcript

        os.remove(compressed_file.name)  # Manually delete the temporary file

    os.remove(input_tempfile.name)  # Manually delete the temporary file
    whisper_cost = duration * 0.006 / 60
    logger.info(f"Whisper cost: {whisper_cost} $")

    total_costs = 0

    # chunk毎に要約を作成
    response_messages, map_costs = ms.map_sammaries(transcript)
    total_costs += map_costs

    logger.info(f"total costs map: {total_costs}")
    logger.info(f"response_messages: {response_messages}")

    doc_summaries: list = response_messages

    output = dict()

    # TPM制限に当たらないように1分待つ
    time.sleep(60)

    def get_simple_summary(doc_summaries: list):
        response, costs = ms.get_simple_summary(doc_summaries)
        logger.info(f"chat create: {response}")
        return response, costs

    simple_summary_response, simple_summary_costs = get_simple_summary(doc_summaries)
    response_message = simple_summary_response["choices"][0]["message"]
    # Step 2: check if GPT wanted to call a function
    if response_message.get("function_call"):
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        try:
            function_args = json.loads(response_message["function_call"]["arguments"])
        except Exception as e:
            logger.error(f"Error parsing function arguments: {e}")
            function_args = None
        simple_summary = function_args
    else:
        simple_summary = None

    output["simple_summary"] = simple_summary
    output["doc_summaries"] = doc_summaries
    output["transcript"] = transcript
    # 結果を加工
    total_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0

    total_tokens += simple_summary_response["usage"]["total_tokens"]
    prompt_tokens += simple_summary_response["usage"]["prompt_tokens"]
    completion_tokens += simple_summary_response["usage"]["completion_tokens"]
    logger.info(f"total tokens: {total_tokens}")

    # gpt-4
    logger.info(f"Whisper cost: {whisper_cost} $")
    logger.info(f"map cost: {map_costs} $")
    logger.info(f"simple summary cost: {simple_summary_costs} $")
    logger.info(f"GPT cost: {map_costs+simple_summary_costs} $")
    logger.info(f"Total cost: {whisper_cost + map_costs+simple_summary_costs} $")

    # Slack通知のブロックを作成
    blocks = []

    # Header
    blocks.append(
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{upload_file.filename}",
                "emoji": True,
            },
        }
    )

    # Summary
    blocks.append(
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Summary:* {output['simple_summary']['summary']}",
            },
        }
    )

    # Bullet Points
    bullet_points_text = "\n".join(
        [f"• {point}" for point in output["simple_summary"]["summary_bullet"]]
    )
    blocks.append(
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Bullet Points:*\n{bullet_points_text}",
            },
        }
    )

    # Decisions
    decisions_text = "\n".join(
        [f"• {decision}" for decision in output["simple_summary"]["decisions"]]
    )
    blocks.append(
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Decisions:*\n{decisions_text}"},
        }
    )

    # Tasks
    tasks_text = "\n".join([f"• {task}" for task in output["simple_summary"]["tasks"]])
    blocks.append(
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Tasks:*\n{tasks_text}"},
        }
    )

    # Long Summary
    long_summary_text = "\n".join(
        [f"• {summary}" for summary in output["doc_summaries"]]
    )
    blocks.append(
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*Long Summary:*\n{long_summary_text}"},
        }
    )

    # Transcriptはスニペットとして表示
    content = output["transcript"]

    # ファイルをアップロード
    response = requests.post(
        "https://slack.com/api/files.upload",
        headers={"Authorization": "Bearer " + slack_token},
        data={
            "channels": "C05TS2WLS74",  # アップロードするチャンネルのID
            "filename": f"transcript_{upload_file.filename}.txt",
            "filetype": "text",
            "content": content,
        },
    )
    logger.info(response.json())

    # レスポンスからファイルのURLを取得
    file_url = response.json().get("file").get("url_private")
    blocks.append(
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Transcript:*\n<{file_url}|transcript.txt>",
            },
        }
    )

    # JSONデータを作成
    json_data = {
        "username": "ボイスレコーダーくん",
        "icon_emoji": ":star2:",
        "blocks": blocks,
    }

    # HTTP POSTリクエストを送信
    response = requests.post(
        WEBHOOK_URL,
        data=json.dumps(json_data),
        headers={"Content-Type": "application/json"},
    )

    # 応答を確認
    if response.status_code != 200:
        raise ValueError(
            f"Request to slack returned an error {response.status_code}, the response is:\n{response.text}"
        )

    # 成功したことをステータスコード200で返す
    return JSONResponse(content=output, status_code=200)


@api_router.post("/summarize")
async def summarize(
    background_tasks: BackgroundTasks,
    upload_file: UploadFile = File(...),
    prompt: Union[str, None] = Query(default=None),
    response_format: Union[str, None] = Query(
        default="text", enum=["text", "vtt", "srt", "verbose_json", "json"]
    ),
):
    background_tasks.add_task(execute_summarize, upload_file, prompt, response_format)

    return {"status": "success", "message": "Message processed successfully."}
