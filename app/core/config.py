from pydantic import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "minutes_generator"
    API_V1_STR: str = "/api/v1"

    class Config:
        # 環境変数のキーの大文字小文字を区別するかどうかを制御します。
        # Trueの場合、環境変数のキーは大文字小文字を区別します。
        case_sensitive = True


settings = Settings()
