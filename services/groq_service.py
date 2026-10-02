import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()


class GroqService:
    """
    Service for Groq API communication.

    The service dynamically discovers available models
    and selects a suitable chat/text generation model.
    """

    PREFERRED_MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b"
    ]


    def __init__(self):

        self.api_key = os.getenv(
            "GROQ_API_KEY"
        )

        if not self.api_key:

            raise RuntimeError(
                "GROQ_API_KEY is missing. "
                "Create a .env file and add your Groq API key."
            )


        self.client = Groq(
            api_key=self.api_key
        )


        self.model = self._discover_model()


    def _discover_model(self):
        """
        Discover an available Groq model.

        Preference is given to known chat-capable models,
        but the API model list is checked first.
        """

        try:

            response = self.client.models.list()

            models = getattr(
                response,
                "data",
                []
            )


            available = []

            for model in models:

                model_id = getattr(
                    model,
                    "id",
                    None
                )

                active = getattr(
                    model,
                    "active",
                    True
                )

                if model_id and active:

                    available.append(
                        model_id
                    )


            # Prefer known suitable models.
            for preferred in self.PREFERRED_MODELS:

                if preferred in available:

                    return preferred


            # Search for likely text/chat models.
            excluded_keywords = [
                "whisper",
                "guard",
                "safeguard",
                "embed",
                "tts",
                "speech",
                "audio"
            ]


            for model_id in available:

                lowered = model_id.lower()

                if any(
                    word in lowered
                    for word in excluded_keywords
                ):

                    continue

                return model_id


            raise RuntimeError(
                "No suitable active Groq chat/text model "
                "was found for this API account."
            )


        except Exception as error:

            # If model listing fails, use a current
            # documented fallback.
            fallback = (
                os.getenv("GROQ_MODEL")
                or "llama-3.3-70b-versatile"
            )

            return fallback


    def get_selected_model(self):

        return self.model


    def generate(
        self,
        system_prompt,
        user_prompt,
        temperature=0.1
    ):
        """
        Generate a response from Groq.
        """

        if not user_prompt.strip():

            raise ValueError(
                "User prompt cannot be empty."
            )


        try:

            completion = (
                self.client
                .chat
                .completions
                .create(
                    model=self.model,

                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ],

                    temperature=temperature
                )
            )


            if not completion.choices:

                raise RuntimeError(
                    "Groq returned no response choices."
                )


            answer = (
                completion
                .choices[0]
                .message
                .content
            )


            if not answer:

                raise RuntimeError(
                    "Groq returned an empty response."
                )


            return answer.strip()


        except Exception as error:

            error_text = str(error)

            if "401" in error_text:

                raise RuntimeError(
                    "Groq authentication failed. "
                    "Please check your GROQ_API_KEY."
                )


            if "429" in error_text:

                raise RuntimeError(
                    "Groq rate limit reached. "
                    "Please wait and try again."
                )


            if "model" in error_text.lower():

                raise RuntimeError(
                    f"Groq model error using "
                    f"'{self.model}': {error_text}"
                )


            raise RuntimeError(
                f"Groq API error: {error_text}"
            )